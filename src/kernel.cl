#define MYFLOAT_MANT_SIZE 28 // WE STORE THE IMPLICIT BIT

typedef struct
{
    bool sign; //TRUE IF NEGATIVE
    int exp;
    bool mant[MYFLOAT_MANT_SIZE];
} myFloat;

myFloat createMyFloat()
{
    myFloat res;

    res.sign = false;
    res.exp = -126;

    for(int i = 0; i < MYFLOAT_MANT_SIZE; ++i) res.mant[i] = false;

    return res;
}

myFloat convertFromFloat(float fNum)
{
    myFloat res = createMyFloat();

    if(fNum == 0)
        return res;

    res.sign = fNum < 0;
    fNum = fNum >= 0 ? fNum : -fNum;

    int intPart = (int)fNum;

    int i;
    for(i = 0; i < MYFLOAT_MANT_SIZE && intPart != 0; ++i)
    {
        res.mant[i] = intPart % 2;
        intPart /= 2;
    }

    //SHIFTING
    for(int j = i - 1; j >= 0; --j)
        res.mant[MYFLOAT_MANT_SIZE - i + j] = res.mant[j];
    
    float fracPart = fNum - (int)fNum;

    int firstOne = -1;

    for (int j = MYFLOAT_MANT_SIZE - i - 1; j >= 0; --j)
    {
        double t = fracPart * 2;
        
        if(t >= 1.0)
        {
            if(firstOne == -1)
                firstOne = j;
                
            fracPart = t - 1.0;
            res.mant[j] = true;
        }
        else
        {
            fracPart = t;
            res.mant[j] = false;
        }
    }

    //SHIFTING WHEN NAT PART IS 0
    if(i == 0)
        for (int j = firstOne; j >= 0; --j)
            res.mant[MYFLOAT_MANT_SIZE - firstOne + j - 1] = res.mant[j];

    res.exp = i > 0 ? (i - 1) : -(MYFLOAT_MANT_SIZE) + firstOne;

    return res;
}

float getAsFloat(myFloat mF)
{
    float res = 0;
    for(int i = 0; i < MYFLOAT_MANT_SIZE; ++i)
        if(mF.mant[i])
            res += pow(2.0, mF.exp - MYFLOAT_MANT_SIZE + 1 + i);

    return (mF.sign ? (-1) : 1) * res;
}

bool exor(bool a, bool b)
{
	return ((a || b) && !(a && b));
}

bool isZero(myFloat mF)
{
	bool b = true;

	for(int i = 0; i < MYFLOAT_MANT_SIZE; ++i)
		b = b && (mF.mant[i] == 0);

	return b;
}

bool isNaN(myFloat mF)
{
	bool b = (mF.exp == 128);
	bool bb = (mF.mant[MYFLOAT_MANT_SIZE - 1] == 1);

    if(!bb)
        return false;

	for(int i = 0; i < MYFLOAT_MANT_SIZE - 1; ++i)
		bb = bb && (mF.mant[i] == 0);

	return (b && !bb);
}

bool isInf(myFloat mF)
{
	if(mF.sign)
		return false;

	bool b = (mF.exp == 128);
	bool bb = (mF.mant[MYFLOAT_MANT_SIZE - 1] == 1);

    if(!bb)
        return false;

	for(int i = 0; i < MYFLOAT_MANT_SIZE - 1; ++i)
		bb = bb && (mF.mant[i] == 0);

	return (b && bb);
}

bool isNegInf(myFloat mF)
{
	if(!mF.sign)
		return false;

	bool b = (mF.exp == 128);
	bool bb = (mF.mant[MYFLOAT_MANT_SIZE - 1] == 1);

    if(!bb)
        return false;

	for(int i = 0; i < MYFLOAT_MANT_SIZE; ++i)
		bb = bb && (mF.mant[i] == 0);

	return (b && bb);
}

bool isDenormal(myFloat mF)
{
	return (mF.exp == -126);
}

int compare(myFloat mF1, myFloat mF2)
{
	//0 - EQUAL
	//-1 - FIRST IS LESS
	//1 - FIRST IS GREATER

	if(mF1.sign && !mF2.sign)
		return -1;
	else if(!mF1.sign && mF2.sign)
		return 1;
	else
	{
		if(mF1.exp < mF2.exp)
			return -1;
		else if(mF1.exp > mF2.exp)
			return 1;
		else
		{
			for(int i = 0; i < MYFLOAT_MANT_SIZE; ++i)
			{
				if(!mF1.mant[MYFLOAT_MANT_SIZE - i - 1] && mF2.mant[MYFLOAT_MANT_SIZE - i - 1])
					return -1;
				else if(mF1.mant[MYFLOAT_MANT_SIZE - i - 1] && !mF2.mant[MYFLOAT_MANT_SIZE - i - 1])
					return 1;
				else
					continue;
			}

			return 0;
		}
	}
}

myFloat addUnsigned(myFloat mF1, myFloat mF2)
{
    if(mF1.exp < mF2.exp)
    {
        myFloat tmp = mF1;
        mF1 = mF2;
        mF2 = tmp;
    }

    myFloat res = createMyFloat();
    uint diff = mF1.exp - mF2.exp;
    bool carry = false;
    bool a = false;
    bool b = false;

    for(int i = 0; i < MYFLOAT_MANT_SIZE; ++i)
    {
        a = exor(mF1.mant[i], carry);
        b = (i + diff < MYFLOAT_MANT_SIZE) ? mF2.mant[i + diff] : false;
        res.mant[i] = exor(a, b);
        carry = (mF1.mant[i] && b) || (mF1.mant[i] && carry) || (b && carry);
    }

    if(carry)
    {
        for(int i = 0; i + 1 < MYFLOAT_MANT_SIZE; ++i)
            res.mant[i] = res.mant[i + 1];

        res.mant[MYFLOAT_MANT_SIZE - 1] = true;
    }

    res.exp = carry ? mF1.exp + 1 : mF1.exp;
    res.sign = false;

    return res;
}

myFloat subtractUnsigned(myFloat mF1, myFloat mF2)
{
    myFloat res = createMyFloat();
    uint diff = mF1.exp - mF2.exp;
    bool carry = false;
    bool b = false;
    bool bCarry = false;
    bool tmp = false;

    for(int i = 0; i < MYFLOAT_MANT_SIZE; ++i)
    {
        tmp = (i + diff < MYFLOAT_MANT_SIZE) ? mF2.mant[i + diff] : false;
        b = exor(tmp, carry);
        bCarry = tmp && carry;
        res.mant[i] = exor(mF1.mant[i], b);
        carry = (!mF1.mant[i] && (tmp || carry)) || (mF1.mant[i] && bCarry);
    }

    int i = MYFLOAT_MANT_SIZE - 1;

    while(i >= 0 && !res.mant[i])
        i--;

    int shift = 0;

    if(i >= 0)
    {
        shift = MYFLOAT_MANT_SIZE - i - 1;

        for(int j = 0; j < i; ++j)
            res.mant[i - j + shift] = res.mant[i - j];

        for(int i = 0; i < shift; ++i)
            res.mant[i] = false;
    }

    res.exp = mF1.exp - shift;

    return res;
}

myFloat subtract(myFloat mF1, myFloat mF2)
{
    if(isZero(mF2))
        return mF1;

    myFloat res = createMyFloat();
    bool prevSign = mF1.sign;

    if(mF1.sign == mF2.sign)
    {
        if(mF1.sign)
        {
            mF1.sign = false;
            mF2.sign = false;
        }

        if(compare(mF1, mF2) == -1) // -2 - -4 -> 2 - 4 = +2
        {
            res = subtractUnsigned(mF2, mF1);
            res.sign = !prevSign;
        }
        else
        {
            res = subtractUnsigned(mF1, mF2); //-4 - (-2) -> 2 - 4 = - 2
            res.sign = prevSign;
        }
    }
    else // -2 - +4 || 2 - - 4 
    {
        mF1.sign = false;
        mF2.sign = false;
        res = addUnsigned(mF1, mF2);
        res.sign = prevSign;
    }

    return res;
}

myFloat add(myFloat mF1, myFloat mF2)
{
    if(isZero(mF1))
        return mF2;
    else if(isZero(mF2))
        return mF1;

    myFloat res = createMyFloat();

    if(!mF1.sign == mF2.sign)
    {
        bool mF1Sign = mF1.sign;
        mF1.sign = false;
        mF2.sign = false;
        res = subtract(mF1, mF2);

        res.sign = mF1Sign ? !res.sign : res.sign;

        return res;
    }

    if(mF1.sign)
    {
        res = addUnsigned(mF1, mF2);
        res.sign = true;
    }
    else
        res = addUnsigned(mF1, mF2);

    return res;
}

myFloat multiply(myFloat mF1, myFloat mF2)
{
    myFloat res = createMyFloat();

    if(isZero(mF1) || isZero(mF2))
        return res;

    bool carry = false;
    int shift = 0;

    for(int i = 0; i < MYFLOAT_MANT_SIZE; ++i)
    {
        if(mF2.mant[MYFLOAT_MANT_SIZE - i - 1])
        {
            myFloat tmp = createMyFloat();

            //COPYING THE MYFLOAT_MANT_SIZE - I DATA FROM THE MF2'S MANTISSA
            for(int j = 0; j < MYFLOAT_MANT_SIZE - i - shift; ++j)
                tmp.mant[j] = mF1.mant[j + i + shift];

            //ADDING ZEROS TO THE TMP'S FRONT (BACK IN OUR STORING)
            for(int j = 0; j < i + shift; ++j)
                tmp.mant[MYFLOAT_MANT_SIZE - i + j] = 0;

            tmp.exp = res.exp;
            res = addUnsigned(res, tmp);
            carry = (tmp.exp != res.exp);
            shift += carry ? 1 : 0;
        }
    }

    res.exp = mF1.exp + mF2.exp + shift;

    res.sign = exor(mF1.sign, mF2.sign);

    return res;
}

__kernel void mandelbrot(__global uchar4 *result, int w, int h, int max_iter, float2 origin, float zoom)
{
    int2 coord = { get_global_id(0), get_global_id(1) };
    int index = coord.y * h + coord.x;

    myFloat cr = convertFromFloat((coord.x / (float)w) - 0.5);
    myFloat ci = convertFromFloat((coord.y / (float)h) - 0.5);

    cr = add(multiply(cr, convertFromFloat(zoom)), convertFromFloat(origin.x));
    ci = add(multiply(ci, convertFromFloat(zoom)), convertFromFloat(origin.y));

    myFloat zr = cr;
    myFloat zi = ci;

    int i = 0;
    for(i = 0; i < max_iter; ++i)
    {
        myFloat real = subtract(multiply(zr, zr), multiply(zi, zi));
        zi = multiply(multiply(convertFromFloat(2), zr), zi);
        zr = real;
        zr = add(zr, cr);
        zi = add(zi, ci);
        
        if(compare(add(multiply(zr, zr), multiply(zi, zi)), convertFromFloat(4)) == 1)
            break;
    }

    float col = i == max_iter ? 0 : i / (float)max_iter;
    float4 color = { col * (i / (255.0f * 255.0f)) * (1.0f / max_iter), col, col, 1.0f };
    uchar4 output = convert_uchar4_sat_rte(color * 255.0f);
    result[index] = output;
}
