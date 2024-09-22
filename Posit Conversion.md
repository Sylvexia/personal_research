
## SoftPosit

```cpp
posit32_t convertDoubleToP32(double f32){

	union ui32_p32 uZ;
	bool sign, regS;
	uint_fast32_t reg, frac=0;
	int_fast32_t exp=0;
	bool bitNPlusOne=0, bitsMore=0;

	(f32>=0) ? (sign=0) : (sign=1);

	if (f32 == 0 ){
		uZ.ui = 0;
		return uZ.p;
	}
	else if(f32 == INFINITY || f32 == -INFINITY || f32 == NAN){
		uZ.ui = 0x80000000;
		return uZ.p;
	}
	else if (f32 == 1) {
		uZ.ui = 0x40000000;
		return uZ.p;
	}
	else if (f32 == -1){
		uZ.ui = 0xC0000000;
		return uZ.p;
	}
	else if (f32 >= 1.329227995784916e+36){
		//maxpos
		uZ.ui = 0x7FFFFFFF;
		return uZ.p;
	}
	else if (f32 <= -1.329227995784916e+36){
		// -maxpos
		uZ.ui = 0x80000001;
		return uZ.p;
	}
	else if(f32 <= 7.52316384526264e-37 && !sign){
		//minpos
		uZ.ui = 0x1;
		return uZ.p;
	}
	else if(f32 >= -7.52316384526264e-37 && sign){
		//-minpos
		uZ.ui = 0xFFFFFFFF;
		return uZ.p;
	}
	else if (f32>1 || f32<-1){
		if (sign){
			//Make negative numbers positive for easier computation
			f32 = -f32;
		}

		regS = 1;
		reg = 1; //because k = m-1; so need to add back 1
		// minpos
		if (f32 <= 7.52316384526264e-37){
			uZ.ui = 1;
		}
		else{
			//regime
			while (f32>=16){
				f32 *=0.0625;  // f32/=16;
				reg++;
			}
			while (f32>=2){
				f32*=0.5;
				exp++;
			}

			int8_t fracLength = 28-reg;

			if (fracLength<0){
				//in both cases, reg=29 and 30, e is n+1 bit and frac are sticky bits
				if(reg==29){
					bitNPlusOne = exp&0x1;
					exp>>=1; //taken care of by the pack algo
				}
				else{//reg=30
					bitNPlusOne=exp>>1;
					bitsMore=exp&0x1;
					exp=0;
				}
				if (f32!=1){//because of hidden bit
					bitsMore =1;
					frac=0;
				}
			}
			else
				frac = convertFractionP32 (f32, fracLength, &bitNPlusOne, &bitsMore);


			if (reg>30 ){
				(regS) ? (uZ.ui= 0x7FFFFFFF): (uZ.ui=0x1);
			}
			//rounding off fraction bits
			else{

				uint_fast32_t regime = 1;
				if (regS) regime = ( (1<<reg)-1 ) <<1;
				if (reg<=28)  exp<<= (28-reg);
				uZ.ui = ((uint32_t) (regime) << (30-reg)) + ((uint32_t) exp ) + ((uint32_t)(frac));
				uZ.ui += (bitNPlusOne & (uZ.ui&1)) | ( bitNPlusOne & bitsMore);
			}
			if (sign) uZ.ui = -uZ.ui & 0xFFFFFFFF;

		}
	}
	else if (f32 < 1 || f32 > -1 ){
		if (sign){
			//Make negative numbers positive for easier computation
			f32 = -f32;
		}
		regS = 0;
		reg = 0;

		//regime
		while (f32<1){
			f32 *= 16;
			reg++;
		}

		while (f32>=2){
			f32*=0.5;
			exp++;
		}


		//only possible combination for reg=15 to reach here is 7FFF (maxpos) and FFFF (-minpos)
		//but since it should be caught on top, so no need to handle
		int_fast8_t fracLength = 28-reg;
		if (fracLength<0){
			//in both cases, reg=29 and 30, e is n+1 bit and frac are sticky bits
			if(reg==29){
				bitNPlusOne = exp&0x1;
				exp>>=1; //taken care of by the pack algo
			}
			else{//reg=30
				bitNPlusOne=exp>>1;
				bitsMore=exp&0x1;
				exp=0;
			}
			if (f32!=1){//because of hidden bit
				bitsMore =1;
				frac=0;
			}
		}
		else
			frac = convertFractionP32 (f32, fracLength, &bitNPlusOne, &bitsMore);


		if (reg>30 ){
			(regS) ? (uZ.ui= 0x7FFFFFFF): (uZ.ui=0x1);
		}
		//rounding off fraction bits
		else{

			uint_fast32_t regime = 1;
			if (regS) regime = ( (1<<reg)-1 ) <<1;
			if (reg<=28)  exp<<= (28-reg);
			uZ.ui = ((uint32_t) (regime) << (30-reg)) + ((uint32_t) exp ) + ((uint32_t)(frac));
			uZ.ui += (bitNPlusOne & (uZ.ui&1)) | ( bitNPlusOne & bitsMore);
		}
		if (sign) uZ.ui = -uZ.ui & 0xFFFFFFFF;

	}
	else {
		//NaR - for NaN, INF and all other combinations
		uZ.ui = 0x80000000;
	}
	return uZ.p;
}
```

## Universal:

```cpp
posit& operator=(float rhs) noexcept {
	return convert_ieee754(rhs);
}
```

```cpp
template <typename T>
constexpr posit<nbits, es>& convert_ieee754(const T& rhs) {
	constexpr int dfbits = std::numeric_limits<T>::digits - 1; // 23
	internal::value<dfbits> v(static_cast<T>(rhs));

	// special case processing
	if (v.iszero()) {
		setzero();
		return *this;
	}
	if (v.isinf() || v.isnan()) {  // posit encode for FP_INFINITE and NaN as NaR (Not a Real)
		setnar();
		return *this;
	}

	convert(v, *this);
	return *this;
}
```

```cpp
// for internal value
	value<fbits>& operator=(float rhs) {
		reset();
		if (_trace_value_conversion) std::cout << "---------------------- CONVERT -------------------" << std::endl;

		switch (std::fpclassify(rhs)) {
		case FP_ZERO:
			_nrOfBits = fbits;
			_zero = true;
			break;
		case FP_INFINITE:
			_inf  = true;
			_sign = true;
			break;
		case FP_NAN:
			_nan = true;
			_sign = true;
			break;
		case FP_SUBNORMAL:
		case FP_NORMAL:
			{
				float _fr{0};
				unsigned int _23b_fraction_without_hidden_bit{0};
				int _exponent{0};
				extract_fp_components(rhs, _sign, _exponent, _fr, _23b_fraction_without_hidden_bit);
				_scale = _exponent - 1;
				_fraction = extract_23b_fraction<fbits>(_23b_fraction_without_hidden_bit);
				_nrOfBits = fbits;
				if (_trace_value_conversion) std::cout << "float " << rhs << " sign " << _sign << " scale " << _scale << " 23b fraction 0x" << std::hex << _23b_fraction_without_hidden_bit << " _fraction b" << _fraction << std::dec << std::endl;
			}
			break;
		}
		return *this;
	}

```

```cpp
// convert a floating point value to a specific posit configuration. Semantically, p = v, return reference to p
template<unsigned nbits, unsigned es, unsigned fbits>
inline posit<nbits, es>& convert(const internal::value<fbits>& v, posit<nbits, es>& p) {
	if constexpr (_trace_conversion) std::cout << "------------------- CONVERT ------------------" << std::endl;
	if constexpr (_trace_conversion) std::cout << "sign " << (v.sign() ? "-1 " : " 1 ") << "scale " << std::setw(3) << v.scale() << " positFraction " << v.positFraction() << std::endl;

	if (v.iszero()) {
		p.setzero();
		return p;
	}
	if (v.isnan() || v.isinf()) {
		p.setnar();
		return p;
	}
	return convert_<nbits, es, fbits>(v.sign(), v.scale(), v.fraction(), p);
}
```

```cpp
template<unsigned nbits, unsigned es, unsigned fbits>
inline posit<nbits, es>& convert_(bool _sign, int _scale, const bitblock<fbits>& positFraction_in, posit<nbits, es>& p) {
	if constexpr (_trace_conversion) std::cout << "------------------- CONVERT ------------------" << std::endl;
	if constexpr (_trace_conversion) std::cout << "sign " << (_sign ? "-1 " : " 1 ") << "scale " << std::setw(3) << _scale << " positFraction " << positFraction_in << std::endl;

	p.clear();
	// construct the posit
	// interpolation rule checks
	if (check_inward_projection_range<nbits, es>(_scale)) {    // positRegime dominated
		if constexpr (_trace_conversion) std::cout << "inward projection" << std::endl;
		// we are projecting to minpos/maxpos
		int k = calculate_unconstrained_k<nbits, es>(_scale);
		k < 0 ? p.setBitblock(minpos_pattern<nbits, es>(_sign)) : p.setBitblock(maxpos_pattern<nbits, es>(_sign));
		// we are done
		if constexpr (_trace_rounding) std::cout << "projection  rounding ";
	}
	else {
		constexpr unsigned pt_len = nbits + 3 + es;
		bitblock<pt_len> pt_bits;
		bitblock<pt_len> positRegime;
		bitblock<pt_len> positExponent;
		bitblock<pt_len> positFraction;
		bitblock<pt_len> sticky_bit;

		bool s = _sign;
		int e  = _scale;
		bool r = (e >= 0);

		unsigned run = unsigned(r ? 1 + (e >> es) : -(e >> es));
		positRegime.set(0, 1 ^ r);
		for (unsigned i = 1; i <= run; i++) positRegime.set(i, r);

		unsigned esval = e % (uint32_t(1) << es);
		positExponent = convert_to_bitblock<pt_len>(esval);
		int nbits_plus_one = static_cast<int>(nbits) + 1;
		int sign_positRegime_es = 2 + int(run) + static_cast<int>(es);
		unsigned nf = (unsigned)std::max<int>(0, (nbits_plus_one - sign_positRegime_es));
		// TODO: what needs to be done if nf > fbits?
		//assert(nf <= input_fbits);
		// copy the most significant nf positFraction bits into positFraction
		unsigned lsb = nf <= fbits ? 0 : nf - fbits;
		for (unsigned i = lsb; i < nf; ++i) positFraction[i] = positFraction_in[static_cast<uint64_t>(fbits) - nf + i];

		bool sb = anyAfter(positFraction_in, static_cast<int64_t>(fbits) - 1ll - static_cast<int64_t>(nf));

		// construct the untruncated posit
		// pt    = BitOr[BitShiftLeft[reg, es + nf + 1], BitShiftLeft[esval, nf + 1], BitShiftLeft[fv, 1], sb];
		positRegime <<= (es + nf + 1ull);
		positExponent <<= (nf + 1ull);
		positFraction <<= 1;
		sticky_bit.set(0, sb);

		pt_bits |= positRegime;
		pt_bits |= positExponent;
		pt_bits |= positFraction;
		pt_bits |= sticky_bit;

		unsigned len = 1 + std::max<unsigned>((nbits + 1), (2 + run + es));
		bool blast = pt_bits.test(len - nbits);
		bool bafter = pt_bits.test(len - nbits - 1);
		bool bsticky = anyAfter(pt_bits, int(len) - static_cast<int>(nbits) - 1 - 1);

		bool rb = (blast & bafter) | (bafter & bsticky);

		bitblock<nbits> ptt;
		pt_bits <<= pt_len - len;
		truncate(pt_bits, ptt);
		if (rb) increment_bitset(ptt);
		if (s) ptt = twos_complement(ptt);
		p.setBitblock(ptt);
	}
	return p;
}

```