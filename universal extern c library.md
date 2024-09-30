
```c
#define POSIT_API POSIT_GLUE(capi, POSIT_NBITS)
```
`capi4`

```cpp
// base functions, e.g.  posit8_t posit8_addp8(posit8_t x, posity_t y)
// these functions must be made specially because everything else is defined in terms of them
#define POSIT_BASE_OP(__rett__, __type__, __op__) \
    __rett__ POSIT_GLUE3(POSIT_MKNAME(__op__),p,POSIT_NBITS)(POSIT_T x, POSIT_T y) POSIT_IMPL({ \
        return POSIT_API::__type__<POSIT_GLUE(op_, __op__)<POSIT_API::nbits, POSIT_API::es>>(x, y); \
    }) \
    POSIT_INLINE(__rett__ POSIT_GLUE3(POSIT_MKNAME(p), POSIT_NBITS, __op__)(POSIT_T x, POSIT_T y) { \
        return POSIT_GLUE3(POSIT_MKNAME(__op__), p, POSIT_NBITS)(x, y); \
    }) \
    POSIT_INLINE(__rett__ POSIT_MKNAME(__op__)(POSIT_T x, POSIT_T y) { \
        return POSIT_GLUE3(POSIT_MKNAME(__op__), p, POSIT_NBITS)(x, y); \
    })
```


```cpp
POSIT_BASE_OP(POSIT_T, op21, add)
```

```bash
nm libposit_c_api_shim.a | grep posit8_add
```

```
0000000000007a90 T posit8_add_exactp8
0000000000027bb0 T posit8_addp8
```

posit8_add() is just a macro that would turn to posit8_addp8()

you can't do this:

```cpp
uint16_t a = 1;
uint16_t d = 2;
pc = posit16_add((posit16_t) a, (posit16_t) d);
```
posit16_add is a union type
we should write our own library now probably

strategy: write wrapper around it:

```cpp
#define OPERATION21(name, ...) \
	template<size_t nbits, size_t es> class name: operation21<nbits,es> { \
		public: static sw::universal::posit<nbits, es> \
			op(sw::universal::posit<nbits, es> a, sw::universal::posit<nbits, es> b) __VA_ARGS__ \
	}
```

```cpp
OPERATION21(op_add, { return a + b; });
OPERATION21(op_sub, { return a - b; });
OPERATION21(op_mul, { return a * b; });
OPERATION21(op_div, { return a / b; });
OPERATION11(op_sqrt, { return sw::universal::sqrt<nbits, es>(a); });
OPERATION11(op_exp, { return sw::universal::exp<nbits, es>(a); });
OPERATION11(op_log, { return sw::universal::log<nbits, es>(a); });
```

sqrt exp log is not implemented in posit, currently it cast to double in universal library.


`nm libposit_c_api_custom.a`

```bash
posit_c_api_custom.cpp.o:
...
0000000000000020 T posit8es0_add
...
0000000000000000 T sylv_test
...
```

negative breaks