***Use Ctrl+SHIFT+V to activate MarkDown Preview Enhanced Window***
==*Ctrl+K,V to open preview to the side*==
1. softmax
    他把一些输入映射为0-1之间的实数，并且归一化保证输出和为1，因此多分类的概率之和也刚好为1。
    输入一个大小为n的一维数组V, $ V_{i} $表示数组第i个元素, 计算后得到的softmax值为:
    $$ S_i = \frac{e^{V_i}}{ \sum_{j=1}^n e^{V_j} } $$
    即该元素的指数与所有元素指数和的比值
---
2. ReLU
   Rectified Linear Unit, 线性整流单元/修正线性单元
   $$ f(x) = \max(0, x) $$
---
3. 