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

probably correct: 
1 110 00 11110000000000000000000000
old universal color print:
1 110 00 00010000000000000000000000
softposit
1 001 11 00010000000000000000000000

`16*(1+0.5+0.25+0.125+0.0625)`