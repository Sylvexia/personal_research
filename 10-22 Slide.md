---
marp: true
theme: default
paginate: true
header: 10-22 Personal Research
footer: 
style: |-
  section {
    font-family: 'Arial', sans-serif;
  }
  pre {
    background-color: #2d2d2d;
    color: #f8f8f2;
    padding: 10px;
    border-radius: 8px;
  }
  h1, h2 {
    color: #2a7ae2;
  }
---

# Software Engineering Concepts  
### by [Your Name]  
#### [Date]  

---

# Agenda  
1. Introduction  
2. Code Examples  
3. Key Concepts  
4. Summary  

---

# Introduction  
- Why is this topic important?  
- What are we going to cover?  

---

# Example Code: Sorting Algorithm  

```javascript
// JavaScript Bubble Sort Example
function bubbleSort(arr) {
  let n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n - i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]]; // Swap
      }
    }
  }
  return arr;
}
console.log(bubbleSort([5, 2, 9, 1, 5, 6]));
