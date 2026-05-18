# thread_block构造函数

## 函数功能

`thread_block`不提供默认的构造函数，用户使用`this_thread_block`函数获取当前线程所属的`thread_block`对象。

## 函数原型

```c++
thread_block this_thread_block()
```

## 参数说明

无

## 返回值说明

返回当前线程所属的`thread_block`对象。

## 约束说明

无

## 调用示例

参考[调用示例](sync.md#调用示例)
