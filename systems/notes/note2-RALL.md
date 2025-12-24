# RALL （Resource Acquisition Is Intialization）

**核心**：资源的生命周期 = 对象的生命周期

- **构造函数**：获取资源
- **析构函数**：释放资源
- 离开作用域时 → 析构函数**自动调用**

## Resources

RALL 不仅释放“内存”，还释放**其他一切资源**

- 内存：new / delete
- 文件：fopen / fclose
- 锁：lock / unlock
- socket：connect / close
- GPU / 设备：init / shutdown
- 事务：begin / commit / rollback

## RALL 的工作机制

- 栈对象作用域结束时必然析构

```
void foo() {
    Resource r;   // 构造：获取资源
    // ...
}              
```

- RAII **通常以类对象的形式存在**，因为 C++ 保证；当对象的生命周期结束（尤其是作用域结束）时，**其析构函数一定会被调用**
- 换句话说：**利用 C++ 对对象生命周期（构造 / 析构）的语言级保证， 将资源管理问题转化为对象生命周期管理问题**

```cpp
FILE* f = fopen("a.txt", "r");
if (!f) return;

// ... 使用 f

fclose(f);
```

- 对于上述例子，我虽然在编码上保证了资源的释放，但是仍然不满足RALL的思想

## RALL的经典标准库实现

- 智能指针

  - `std::unique_ptr`

  - `std::shared_ptr`

```
std::unique_ptr<Foo> p = std::make_unique<Foo>();
```

- 容器

  - `std::vector`

  - `std::string`

  - `std::map`

- 锁

  - `std::lock_guard`

  - `std::unique_lock`

- 文件流

  - `std::ifstream`

  - `std::ofstream`

## 栈对象

**栈对象**

```
void foo() {
    Foo a;
}
```

- 析构时机：**离开作用域立即析构**
- RAII 最理想的载体

**堆对象**：

```
Foo* p = new Foo();
delete p;
```

- 对象本身在堆上
- 生命周期由 `new / delete` 决定
- **不会自动析构**；极易泄漏

**静态对象**：

- 函数内静态

```
void foo() {
    static Foo s;
}
```

- 全局 / 命名空间静态

```
Foo g;
```

- 生命周期：程序开始 → 程序结束
- 析构：**程序退出时**

**成员对象**

```
struct A {
    Foo f;
};
```

- 析构时机： **随所属对象一起析构**

**临时对象（Temporary object)**

```cpp
Foo f = Foo();
```

- 生命周期通常是：表达式结束

- 受优化（RVO / NRVO）影响

- 析构一定发生，但时机不总是直观

**智能指针管理的对象**

```
std::unique_ptr<Foo> p = std::make_unique<Foo>();
```

- `Foo` 在堆上
- `unique_ptr` 是栈对象（或成员 / 静态）
- **真正 RAII 的是 `unique_ptr`**

## 构造函数和析构函数

- 构造函数是在对象“被创建时”自动调用的特殊成员函数， 用来完成对象的初始化与资源获取

- 构造函数需要**成员初始化列表**：

  - `const` 成员
  - 引用成员
  - 没有默认构造函数的成员
  - **性能更好（直接构造，而非先默认构造再赋值）**

- 析构函数是在对象“生命周期结束时”自动调用的特殊成员函数，用来完成资源释放和清理工作

  - 没有参数
  - 没有返回值
  - **每个对象只调用一次**
  - 析构函数 **不应抛异常**

- 构造 / 析构的调用顺序：**后构造的，先析构**

  ```
  {
      A a;
      B b;
  }
  ```

  顺序：

  ```
  A 构造
  B 构造
  B 析构
  A 析构
  ```

- 在继承关系中：

  ```
  Base 构造 → Derived 构造
  Derived 析构 → Base 析构
  ```

