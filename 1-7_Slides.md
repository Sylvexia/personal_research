---
marp: true
theme: default
paginate: true
header: 
footer: 
style: "h1, h2, h3 {\r  text-align: center;\r}

  pre, code {\r  background-color: #ffffff;\r    \r  color: #2d2d2d; \r  \r  font-size: auto;\r }\r

  section {\r  font-size: auto;\r}\r

  img[alt~=\"center\"]\ 

  {\r  display: block;\r  margin: 0 auto;\r}"

---

# 1-7 Personal Research
## Presenter: Yu Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Simplify and find what should we do next.
```bash
git diff --shortstat
6 files changed, 26 insertions(+), 351 deletions(-)
```

---

# What we should do next?

1. Run and execute at least one more model. e.g. gpt-2
2. posit math operation support. e.g. `exp`, `sqrt`, `tanh`, `erf`
3. Migrate the logic such that orthogonal to the original project
	- For future implementation like posit Dialect.
4. Good numerical metric for posit operation.