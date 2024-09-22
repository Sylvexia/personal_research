
Posit_Add(a, b):

if(a is nar() or b is nar())
return nar();

if(a is zero)
return b;

if(b is zero)
return a;

// compile time given nbits and es bit
fbits = (es + 2 >= nbits ? 0 : nbits - 3 - es);  // max posit fraction bits
fhbit = fbit + 1;
abits = fhbit+3 // size of addend

let sum[abits + 1] sum
let a[fbits] b[fbits]

normalize(a)
normalize(b)

// normalize

//populate positRegime positExponent positFraction

decode()
// decode takes the raw bits representing a posit coming from memory
// and decodes the sign, positRegime, the positExponent, and the positFraction.

module_add<fbits, abits>(a, b, sum)

if(a is inf or b is inf)
return inf

arith.lsb as
asdfasj