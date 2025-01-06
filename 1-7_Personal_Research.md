
# Todo

- Refactor our pass and slowly make the logic orthogonal to the project
- Revise our conversion scheme.
- Custom attribute.
- Posit Dialect.
- What does vector Dialect do

---
# Summary

- Simplify and find what should we do next.
```bash
git diff --shortstat
 5 files changed, 24 insertions(+), 349 deletions(-)
```

---

# What we should do next?

1. Run and execute at least one more model. e.g. gpt-2
2. posit math operation support. e.g. `exp`, `sqrt`, `tanh`, `erf`
3. Migrate the logic such that orthogonal to the original project
	- For future implementation like Posit Dialect.