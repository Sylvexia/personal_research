洪祐鈞

# TODO

- Math Dialect lowering.
	- sitofp, exp, sqrt, tanh, erf
		- wrapper
		- mlir lowering
	- sitofp is arith
- Runtime
- Custom attribute.
- Posit Dialect.
- What does vector Dialect do

babylonian
```
template<unsigned nbits, unsigned es, unsigned fbits>
	inline internal::value<fbits> fast_sqrt(internal::value<fbits>& v) {
		if (_trace_sqrt) std::cout << "---------------------------  SQRT -----------------------" << std::endl;
		//			static_assert(nbits >= 16, "fast_sqrt requires posit configurations nbits >= 16");
		posit<nbits, es> fr = v.fraction_value()*0.5;
		int e = v.scale() + 1;
		posit<nbits, es> y = posit<nbits, es>(0.41731f) + posit<nbits, es>(0.59016f) * fr;
		posit<nbits, es> z = y + fr / y;
		if (_trace_sqrt) {
			std::cout << "f          " << v << std::endl;
			std::cout << "e          " << e << std::endl;
			std::cout << "fr         " << fr << std::endl;
			std::cout << "y0         " << y << std::endl;
			std::cout << "y1         " << z << std::endl;
		}
		y = posit<nbits, es>(0.25f) * z + fr / z;
		if (_trace_sqrt) std::cout << "y2         " << y << std::endl;

		if (e % 2) {
			y *= posit<nbits, es>(0.707106781186547524400844362104);
			if (_trace_sqrt) std::cout << "y*sqrt0.5  " << y << std::endl;
			y = (y < posit<nbits, es>(0.5f) ? posit<nbits, es>(0.5f) : y);
			e += 1;
		}
		else {
			posit<nbits, es> one(1.0f), onemme;
			onemme = --one;
			y = (y < one ? y : onemme);
		}
		if (_trace_sqrt) std::cout << "y adjusted " << y << std::endl;

		internal::value<fbits> vsqrt = y.to_value();
		vsqrt.setscale((e >> 1) - 1);
		if (_trace_sqrt) std::cout << "vsqrt      " << vsqrt << std::endl;
		return vsqrt;
	}
```

Example: `exp(x)` is computed using `x=nln⁡(2)+rx = n \ln(2) + rx=nln(2)+r`, where r is small.

# issue

Testing posit8es2
`FAIL: testInput = 11110101 positResultRaw = 00000001 doubleResultRaw = 00000000 doubleValue = -1536 doubleResult = 0`

```
// Base-e exponential function
template<unsigned nbits, unsigned es>
posit<nbits,es> exp(posit<nbits,es> x) {
	if (isnar(x)) return x;
	posit<nbits, es> p;
	double d = std::exp(double(x));
	if (d == 0.0) {
		p.minpos();//should be zero?
	}
	else {
		p = d;
	}
	return p;
}
```

posit16es2 
FAIL: a = 62264 ra = 1 ra_ref = 0
Passed: 62173 Failed: 3363

