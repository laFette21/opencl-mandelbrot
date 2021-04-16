#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <OpenGL/OpenGL.h>
#include <OpenCL/opencl.h>
#include <GLUT/glut.h>

#include <mach/mach_time.h>

////////////////////////////////////////////////////////////////////////////////

#define USE_GL_ATTACHMENTS              (1)
#define DEBUG_INFO                      (0)     
#define COMPUTE_KERNEL_FILENAME         ("kernel.cl")
#define COMPUTE_KERNEL_METHOD_NAME      ("mandelbrot")
#define SEPARATOR                       ("----------------------------------------------------------------------\n")
#define WIDTH                           (512)
#define HEIGHT                          (512)

////////////////////////////////////////////////////////////////////////////////

static cl_context                       ComputeContext;
static cl_command_queue                 ComputeCommands;
static cl_kernel                        ComputeKernel;
static cl_program                       ComputeProgram;
static cl_device_id                     ComputeDeviceId;
static cl_device_type                   ComputeDeviceType;
static cl_mem                           ComputeResult;
static cl_mem                           ComputeImage;
static size_t                           MaxWorkGroupSize;
static int                              WorkGroupSize[2];
static int                              WorkGroupItems = 32;

////////////////////////////////////////////////////////////////////////////////

static int Width                        = WIDTH;
static int Height                       = HEIGHT;

static int Update                       = 1;
static int MaxIterations                = 50;

static float Origin[2]                  = {-0.75, 0};
static float Zoom                       = 3.0f;

////////////////////////////////////////////////////////////////////////////////

static uint TextureId                   = 0;
static uint TextureTarget               = GL_TEXTURE_2D;
static uint TextureInternal             = GL_RGBA;
static uint TextureFormat               = GL_RGBA;
static uint TextureType                 = GL_UNSIGNED_BYTE;
static uint TextureWidth                = WIDTH;
static uint TextureHeight               = HEIGHT;
static size_t TextureTypeSize           = sizeof(char);
static uint ActiveTextureUnit           = GL_TEXTURE1_ARB;
static void* HostImageBuffer            = 0;

static double TimeElapsed               = 0;
static int FrameCount                   = 0;
static uint ReportStatsInterval         = 30;

static char StatsString[512]            = "\0";

static float VertexPos[4][2]            = { { -1.0f, -1.0f },
                                            { +1.0f, -1.0f },
                                            { +1.0f, +1.0f },
                                            { -1.0f, +1.0f } };
static float TexCoords[4][2];

////////////////////////////////////////////////////////////////////////////////

static int DivideUp(int a, int b) 
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

static uint64_t GetCurrentTime()
{
    return mach_absolute_time();
}
	
static double SubtractTime( uint64_t uiEndTime, uint64_t uiStartTime )
{    
	static double s_dConversion = 0.0;
	uint64_t uiDifference = uiEndTime - uiStartTime;
	if( 0.0 == s_dConversion )
	{
		mach_timebase_info_data_t kTimebase;
		kern_return_t kError = mach_timebase_info( &kTimebase );
		if( kError == 0  )
			s_dConversion = 1e-9 * (double) kTimebase.numer / (double) kTimebase.denom;
    }
		
	return s_dConversion * (double) uiDifference; 
}

////////////////////////////////////////////////////////////////////////////////

static int LoadTextFromFile(const char *file_name, char **result_string, size_t *string_len)
{
    int fd;
    unsigned file_len;
    struct stat file_status;
    int ret;

    *string_len = 0;
    fd = open(file_name, O_RDONLY);
    if (fd == -1)
    {
        printf("Error opening file %s\n", file_name);
        return -1;
    }
    ret = fstat(fd, &file_status);
    if (ret)
    {
        printf("Error reading status for file %s\n", file_name);
        return -1;
    }
    file_len = file_status.st_size;

    *result_string = (char*)calloc(file_len + 1, sizeof(char));
    ret = read(fd, *result_string, file_len);
    if (!ret)
    {
        printf("Error reading from file %s\n", file_name);
        return -1;
    }

    close(fd);

    *string_len = file_len;
    return 0;
}

static void CreateTexture(uint width, uint height)
{    
    if(TextureId)
        glDeleteTextures(1, &TextureId);
    TextureId = 0;
    
    printf("Creating Texture %d x %d...\n", width, height);

    TextureWidth = width;
    TextureHeight = height;
    
    glActiveTextureARB(ActiveTextureUnit);
    glGenTextures(1, &TextureId);
    glBindTexture(TextureTarget, TextureId);
    glTexParameteri(TextureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(TextureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(TextureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(TextureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(TextureTarget, 0, TextureInternal, TextureWidth, TextureHeight, 0, 
                 TextureFormat, TextureType, 0);
    glBindTexture(TextureTarget, 0);
}

static void RenderTexture( void *pvData )
{
    glDisable( GL_LIGHTING );

    glViewport( 0, 0, Width * 2, Height * 2 );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluOrtho2D( -1.0, 1.0, -1.0, 1.0 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glMatrixMode( GL_TEXTURE );
    glLoadIdentity();

    glEnable( TextureTarget );
    glBindTexture( TextureTarget, TextureId );

    if(pvData)
        glTexSubImage2D(TextureTarget, 0, 0, 0, TextureWidth, TextureHeight, 
                        TextureFormat, TextureType, pvData);

    glTexParameteri(TextureTarget, GL_TEXTURE_COMPARE_MODE_ARB, GL_NONE);
    glBegin( GL_QUADS );
    {
        glColor3f(1.0f, 1.0f, 1.0f);
        glTexCoord2f( 0.0f, 0.0f );
        glVertex3f( -1.0f, -1.0f, 0.0f );

        glTexCoord2f( 0.0f, 1.0f );
        glVertex3f( -1.0f, 1.0f, 0.0f );

        glTexCoord2f( 1.0f, 1.0f );
        glVertex3f( 1.0f, 1.0f, 0.0f );

        glTexCoord2f( 1.0f, 0.0f );
        glVertex3f( 1.0f, -1.0f, 0.0f );
    }
    glEnd();
    glBindTexture( TextureTarget, 0 );
    glDisable( TextureTarget );
}

static int Recompute(void)
{
    if(!ComputeKernel || !ComputeResult)
        return CL_SUCCESS;
        
    void *values[10];
    size_t sizes[10];
    size_t global[2];
    size_t local[2];

    int err = 0;
    unsigned int v = 0, s = 0, a = 0;
    
    values[v++] = &ComputeResult;
    values[v++] = &Width;
    values[v++] = &Height;
    values[v++] = &MaxIterations;
    values[v++] = &Origin;
    values[v++] = &Zoom;

    sizes[s++] = sizeof(cl_mem);
    sizes[s++] = sizeof(int);
    sizes[s++] = sizeof(int);
    sizes[s++] = sizeof(int);
    sizes[s++] = (2 * sizeof(float));
    sizes[s++] = sizeof(float);

    if(Update)
    {
        Update = 0;
        err = CL_SUCCESS;
        for (a = 0; a < s; a++)
            err |= clSetKernelArg(ComputeKernel, a, sizes[a], values[a]);
    
        if (err)
            return -10;
    }
    
    int size_x = WorkGroupSize[0];
    int size_y = WorkGroupSize[1];
    
    global[0] = DivideUp(TextureWidth, size_x) * size_x; 
    global[1] = DivideUp(TextureHeight, size_y) * size_y;
    
    local[0] = size_x;
    local[1] = size_y;

#if (DEBUG_INFO)
    if(FrameCount <= 1)
        printf("Global[%4d %4d] Local[%4d %4d]\n", 
            (int)global[0], (int)global[1],
            (int)local[0], (int)local[1]);
#endif

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel, 2, NULL, global, local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to enqueue kernel! %d\n", err);
        return err;
    }

#if (USE_GL_ATTACHMENTS)

    err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &ComputeImage, 0, 0, 0);
    if (err != CL_SUCCESS)
    {
        printf("Failed to acquire GL object! %d\n", err);
        return EXIT_FAILURE;
    }

    size_t origin[] = { 0, 0, 0 };
    size_t region[] = { TextureWidth, TextureHeight, 1 };
    err = clEnqueueCopyBufferToImage(ComputeCommands, ComputeResult, ComputeImage, 
                                     0, origin, region, 0, NULL, 0);
    
    if(err != CL_SUCCESS)
    {
        printf("Failed to copy buffer to image! %d\n", err);
        return EXIT_FAILURE;
    }
    
    err = clEnqueueReleaseGLObjects(ComputeCommands, 1, &ComputeImage, 0, 0, 0);
    if (err != CL_SUCCESS)
    {
        printf("Failed to release GL object! %d\n", err);
        return EXIT_FAILURE;
    }

#else

    err = clEnqueueReadBuffer( ComputeCommands, ComputeResult, CL_TRUE, 0, Width * Height * TextureTypeSize * 4, HostImageBuffer, 0, NULL, NULL );      
    if (err != CL_SUCCESS)
    {
        printf("Failed to read buffer! %d\n", err);
        return EXIT_FAILURE;
    }

#endif

    return CL_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////

static int CreateComputeResult(void)
{
    int err = 0;
        
#if (USE_GL_ATTACHMENTS)

    if(ComputeImage)
        clReleaseMemObject(ComputeImage);
    ComputeImage = 0;
    
    printf("Allocating compute result image in device memory...\n");
    ComputeImage = clCreateFromGLTexture2D(ComputeContext, CL_MEM_WRITE_ONLY, TextureTarget, 0, TextureId, &err);
    if (!ComputeImage || err != CL_SUCCESS)
    {
        printf("Failed to create OpenGL texture reference! %d\n", err);
        return -1;
    }

#else

    if (HostImageBuffer)
        free(HostImageBuffer);

    printf("Allocating compute result image in host memory...\n");
    HostImageBuffer = malloc(TextureWidth * TextureHeight * TextureTypeSize * 4);
    if(!HostImageBuffer)
    {
        printf("Failed to create host image buffer!\n");
        return -1;
    }
     
    memset(HostImageBuffer, 0, TextureWidth * TextureHeight * TextureTypeSize * 4);

#endif

    if(ComputeResult)
        clReleaseMemObject(ComputeResult);
    ComputeResult = 0;
    
    ComputeResult = clCreateBuffer(ComputeContext, CL_MEM_WRITE_ONLY, TextureTypeSize * 4 * TextureWidth * TextureHeight, NULL, NULL);
    if (!ComputeResult)
    {
        printf("Failed to create OpenCL array!\n");
        return -1;
    }

    return CL_SUCCESS;
}

static int SetupComputeDevices(int gpu)
{
    int err;
	size_t returned_size;
    ComputeDeviceType = gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

#if (USE_GL_ATTACHMENTS)

    printf(SEPARATOR);
    printf("Using active OpenGL context...\n");

    CGLContextObj kCGLContext = CGLGetCurrentContext();          
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
    
    cl_context_properties properties[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        (cl_context_properties)kCGLShareGroup, 0
    };
        
    // Create a context from a CGL share group
    //
    ComputeContext = clCreateContext(properties, 0, 0, clLogMessagesToStdoutAPPLE, 0, 0);
    if (!ComputeContext)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

#else

    // Locate a compute device
    //
    err = clGetDeviceIDs(NULL, ComputeDeviceType, 1, &ComputeDeviceId, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to locate compute device!\n");
        return EXIT_FAILURE;
    }
  
    // Create a context containing the compute device(s)
    //
    ComputeContext = clCreateContext(0, 1, &ComputeDeviceId, clLogMessagesToStdoutAPPLE, NULL, &err);
    if (!ComputeContext)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

#endif

    unsigned int device_count;
    cl_device_id device_ids[16];

    err = clGetContextInfo(ComputeContext, CL_CONTEXT_DEVICES, sizeof(device_ids), device_ids, &returned_size);
    if(err)
    {
        printf("Error: Failed to retrieve compute devices for context!\n");
        return EXIT_FAILURE;
    }
    
    device_count = returned_size / sizeof(cl_device_id);
    
    int i = 0;
    int device_found = 0;
    cl_device_type device_type;	
    for(i = 0; i < device_count; i++) 
    {
        clGetDeviceInfo(device_ids[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
        if(device_type == ComputeDeviceType) 
        {
            ComputeDeviceId = device_ids[i];
            device_found = 1;
            break;
        }	
    }
    
    if(!device_found)
    {
        printf("Error: Failed to locate compute device!\n");
        return EXIT_FAILURE;
    }
        
    // Create a command queue
    //
    ComputeCommands = clCreateCommandQueue(ComputeContext, ComputeDeviceId, 0, &err);
    if (!ComputeCommands)
    {
        printf("Error: Failed to create a command queue!\n");
        return EXIT_FAILURE;
    }

    // Report the device vendor and device name
    // 
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    err = clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
    err|= clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve device info!\n");
        return EXIT_FAILURE;
    }

    printf(SEPARATOR);
    printf("Connecting to %s %s...\n", vendor_name, device_name);

    return CL_SUCCESS;
}

static int SetupComputeKernel(void)
{
    int err = 0;
    char *source = 0;
    size_t length = 0;

    if(ComputeKernel)
        clReleaseKernel(ComputeKernel);    
    ComputeKernel = 0;

    if(ComputeProgram)
        clReleaseProgram(ComputeProgram);
    ComputeProgram = 0;
    
    printf(SEPARATOR);
    printf("Loading kernel source from file '%s'...\n", COMPUTE_KERNEL_FILENAME);    
    err = LoadTextFromFile(COMPUTE_KERNEL_FILENAME, &source, &length);
    if (!source || err)
    {
        printf("Error: Failed to load kernel source!\n");
        return EXIT_FAILURE;
    }

#if (DEBUG_INFO)
    printf("%s\n", source);
#endif

    // Create the compute program from the source buffer
    //
    ComputeProgram = clCreateProgramWithSource(ComputeContext, 1, (const char **) &source, NULL, &err);
    if (!ComputeProgram || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    free(source);

    // Build the program executable
    //
    err = clBuildProgram(ComputeProgram, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(ComputeProgram, ComputeDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from within the program
    //
    printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME);    
    ComputeKernel = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME, &err);
    if (!ComputeKernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return EXIT_FAILURE;
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &MaxWorkGroupSize, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

#if (DEBUG_INFO)
    printf("MaxWorkGroupSize: %d\n", MaxWorkGroupSize);
    printf("WorkGroupItems: %d\n", WorkGroupItems);
#endif

    WorkGroupSize[0] = (MaxWorkGroupSize > 1) ? (MaxWorkGroupSize / WorkGroupItems) : MaxWorkGroupSize;
    WorkGroupSize[1] = MaxWorkGroupSize / WorkGroupSize[0];

    printf(SEPARATOR);

    return CL_SUCCESS;

}

static void Cleanup(void)
{
    clFinish(ComputeCommands);
    clReleaseKernel(ComputeKernel);
    clReleaseProgram(ComputeProgram);
    clReleaseCommandQueue(ComputeCommands);
    clReleaseMemObject(ComputeResult);
    clReleaseMemObject(ComputeImage);
    clReleaseContext(ComputeContext);
    
    ComputeCommands = 0;
    ComputeKernel = 0;
    ComputeProgram = 0;    
    ComputeResult = 0;
    ComputeImage = 0;
    ComputeContext = 0;
}

static void Shutdown(void)
{
    printf(SEPARATOR);
    printf("Shutting down...\n");
    Cleanup();
    exit(0);
}

////////////////////////////////////////////////////////////////////////////////

static int SetupGraphics(void)
{
    CreateTexture(Width, Height);

    glClearColor (0.0, 0.0, 0.0, 0.0);

    glDisable(GL_DEPTH_TEST);
    glActiveTexture(GL_TEXTURE0);
    glViewport(0, 0, Width, Height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    TexCoords[3][0] = 0.0f;
    TexCoords[3][1] = 0.0f;
    TexCoords[2][0] = Width;
    TexCoords[2][1] = 0.0f;
    TexCoords[1][0] = Width;
    TexCoords[1][1] = Height;
    TexCoords[0][0] = 0.0f;
    TexCoords[0][1] = Height;

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, VertexPos);
    glClientActiveTexture(GL_TEXTURE0);
    glTexCoordPointer(2, GL_FLOAT, 0, TexCoords);
    return GL_NO_ERROR;
}

static int Initialize(int gpu)
{
    int err;
    err = SetupGraphics();
    if (err != GL_NO_ERROR)
    {
        printf ("Failed to setup OpenGL state!");
        exit (err);
    }

    err = SetupComputeDevices(gpu);
    if(err != CL_SUCCESS)
    {
        printf ("Failed to connect to compute device! Error %d\n", err);
        exit (err);
    }

    cl_bool image_support;
    err = clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_IMAGE_SUPPORT,
                          sizeof(image_support), &image_support, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to query device for image support");
        exit(err);
    }
    if (image_support == CL_FALSE) {
        printf("Qjulia requires images: Images not supported on this device.");
        return CL_IMAGE_FORMAT_NOT_SUPPORTED;
    }
    
    err = SetupComputeKernel();
    if (err != CL_SUCCESS)
    {
        printf ("Failed to setup compute kernel! Error %d\n", err);
        exit (err);
    }
    
    err = CreateComputeResult();
    if(err != CL_SUCCESS)
    {
        printf ("Failed to create compute result! Error %d\n", err);
        exit (err);
    }
    
    return CL_SUCCESS;
}

static void ReportStats(uint64_t uiStartTime, uint64_t uiEndTime)
{
    TimeElapsed += SubtractTime(uiEndTime, uiStartTime);

    if(TimeElapsed && FrameCount && FrameCount > ReportStatsInterval) 
	{
        double fMs = (TimeElapsed * 1000.0 / (double) FrameCount);
        double fFps = 1.0 / (fMs / 1000.0);
        
        sprintf(StatsString, "[%s] Compute: %3.2f ms Display: %3.2f fps (%s) Zoom: %f Position: (%f, %f)\n", 
                (ComputeDeviceType == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU", 
                fMs, fFps, USE_GL_ATTACHMENTS ? "attached" : "copying", Zoom, Origin[0], Origin[1]);
		
		glutSetWindowTitle(StatsString);

		FrameCount = 0;
        TimeElapsed = 0;
	}    
}

static void Display(void)
{
    FrameCount++;
    uint64_t uiStartTime = GetCurrentTime();
    
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glClear (GL_COLOR_BUFFER_BIT);
    
    int err = Recompute();
    if (err != 0)
    {
        printf("Error %d from Recompute!\n", err);
        exit(1);
    }

    RenderTexture(HostImageBuffer);
    
    glFinish(); // for timing
    
    uint64_t uiEndTime = GetCurrentTime();
    ReportStats(uiStartTime, uiEndTime);
    glutSwapBuffers();
}

void Keyboard(unsigned char key, int x, int y)
{
    const float move_speed = 0.05f;
    const float zoom_speed = 1.05f;

    switch( key )
    {
        case 27:
            exit(0);
            break;

        case 'a':
            Origin[0] -= move_speed * Zoom;
            break;

        case 'd':
            Origin[0] += move_speed * Zoom;
            break;

        case 's':
            Origin[1] -= move_speed * Zoom;
            break;

        case 'w':
            Origin[1] += move_speed * Zoom;
            break;

        case 'z':
            Zoom /= zoom_speed;
            break;

        case 'u':
            Zoom *= zoom_speed;
            break;

        case '0':
            Zoom = 3;
            Origin[0] = -0.75;
            Origin[1] = 0;
            break;

        case '1':
            Zoom = 0.000005;
            Origin[0] = 0.241550;
            Origin[1] = 0.568976;
            break;

        case '2':
            Zoom = 0.000009;
            Origin[0] = 0.347425;
            Origin[1] = -0.581360;
            break;
        
        case '3':
            Zoom = 0.000005;
            Origin[0] = -1.942068;
            Origin[1] = 0.000409;
            break;

        case '4':
            Zoom = 0.000005;
            Origin[0] = -0.786518;
            Origin[1] = 0.165409;
            break;
        
        case '5':
            Zoom = 0.000005;
            Origin[0] = -0.742016;
            Origin[1] = 0.245320;
            break;

        case 'f':
            glutFullScreen(); 
            break;

    }

    Update = 1;
    glutPostRedisplay();
}

void Idle(void)
{
    glutPostRedisplay();
}

int main(int argc, char** argv)
{
    // Parse command line options
    //
    int i;
    int use_gpu = 1;
    for(i = 0; i < argc && argv; i++)
    {
        if(!argv[i])
            continue;
            
        if(strstr(argv[i], "cpu"))
            use_gpu = 0;        

        else if(strstr(argv[i], "gpu"))
            use_gpu = 1;
    }
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(Width, Height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(argv[0]);

    if (Initialize(use_gpu) == GL_NO_ERROR)
    {
        glutDisplayFunc(Display);
        glutIdleFunc(Idle);
        glutKeyboardFunc(Keyboard);

        atexit(Shutdown);
        printf("Starting event loop...\n");

        glutMainLoop();
    }

    return 0;
}
