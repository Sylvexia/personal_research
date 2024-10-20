-31
Universal library:

```bash
(base) sylvex@sylvex-Aspire-A715-51G:~/universal/build_clang/tools/cmd$ ./posit -31
 posit<32,2>  = 11100000010000000000000000000000 : -31


Different printing formats
floating-point : -31
triple form    : (-, 4, 1111000000000000000000000000000000000000000000000000000000)
binary form    : 0b1.110.00.11'1100'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000
pretty print   : s1 r110 e00 f1111000000000000000000000000000000000000000000000000000000 qNW v-31
color coded    : 1110000001000000000000000000000000000000000000000000000000000000
hex print      : 64.2x0x09c40000000000000p
```

SoftPosit

```cpp
pZ = convertFloatToP32(-31);

double dZ = convertP32ToDouble(pZ);

printf("dZ: %.15f\n", dZ);

uint32_t uiZ = castUI(pZ);

printBinary((uint64_t*)&uiZ, 32);
```

```bash
(base) sylvex@sylvex-Aspire-A715-51G:~/posit_exp$ ./main 
dZ: -31.000000000000000
 10011100 01000000 00000000 00000000
```

posit<32, 2>
probably correct: 
1 110 00 11110000000000000000000000
old universal color print:
1 110 00 00010000000000000000000000
softposit
1 001 11 00010000000000000000000000

`(2^(2^2))*(1+0.5+0.25+0.125+0.0625) = 31`


posit< 8,1>  = 01110110 : 48
posit< 8,1>  = 01110111 : 56
posit< 8,1>  = 01111000 : 64

posit< 8,1>  = 01111010 : 128
posit< 8,1>  = 01111011 : 192
posit< 8,1>  = 01111100 : 256

posit< 8,2> = 
0 110 11 10 : 192
0 110 11 11 : 224
0 1110 00 0 : 256

0 001 11 10: `2^(2^-2) * 2^3 * 1.5`
0 001 11 11: `2^(2^-2) * 2^3 * 1.75`

0 0001 00 0: `2^(2^-3) * 2^2 * 1`
or
0 01 00 000: `2^(2^-1) * 2^0 * 1`

posit< 8,3>  = 01010010 : 24
posit< 8,3>  = 01010011 : 28
posit< 8,3>  = 01010100 : 32