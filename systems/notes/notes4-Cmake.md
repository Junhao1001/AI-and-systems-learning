# Cmake

## Concepts

- **目录作用域（Directory Scope)**: 
  - 由一个 `CMakeLists.txt` 文件所形成的变量与命令生效范围
  - 只要有一个 `CMakeLists.txt`，**它所在的目录就是一个作用域节点**
  - 目录作用域的作用：
    - 组织工程结构
    - 控制变量污染范围

## Cmake的层级

在 CMake 里，**最核心的对象是target(目标）**；

**CMake = 在不同层级上“定义 target + 给 target 赋属性”**

Cmake一般涉及4个概念层级：

```
Workspace（源码树 / 构建树）
 └── Project
      └── Directory
           └── Target
                └── Source / Property
```

### Project

project 是一个“逻辑工程单位”

```cmake
project(MyProject LANGUAGES C CXX)
```

主要完成一下工作：

- 定义工程名
- 定义支持的语言（C / CXX / CUDA / ASM）
- 初始化一组变量：
  - `PROJECT_NAME`
  - `PROJECT_SOURCE_DIR`
  - `PROJECT_BINARY_DIR`
- 设置全局编译器相关信息

project可以设置多个，但是大多数工程只需要一个顶层project

### Directory

**每个包含 `CMakeLists.txt` 的目录，都是一个 directory scope**

```
add_subdirectory(lib)
```

就会创建一个新的 **目录作用域**

- 有作用域继承
- 可以定义变量
- 调用`add_library`/`add_executable`
- 但 **directory 本身不是构建对象**, 只是一个**组织和作用域单元**

### Target

**target = 最终参与构建的实体**

例如：

```
add_library(mathlib ...)
add_executable(app ...)
```

创建的 `mathlib`、`app` **都是 target**

- **target 属于 Directory**，但生命周期跨 Directory
- 换句话说，**target定义在某个目录，但是可以被全局引用**

```cmake
# lib/CMakeLists.txt
add_library(mathlib ...)

# app/CMakeLists.txt
target_link_libraries(app mathlib)
```

**target的本质**：一组属性的集合

- 源文件
- include路径
- 宏定义
- 编译选项
- 链接库
- 输出类型（exe/static/shared)

**library和executable的区别？**

- **executable:**

  ```
  add_executable(app main.cpp)
  ```

  - 生成可执行文件
  - 有入口点`main`

- **library**:

  ```
  add_library(mylib STATIC a.cpp)
  ```

  - 生成库
  - **不能单独运行**
  - 被其他 target 链接
  - library是**“属性载体”**（可以被`target_link_libraries`使用），executable 不是

#### INTERFACE target

**现代 CMake 依赖管理的基石**

- 不生成文件

- 只携带属性

- 专门用来“传播 include / definitions”

```
add_library(header_only INTERFACE)
```

#### IMPORTED target (之后展开学习)

```
add_library(OpenSSL::SSL IMPORTED)
```

- 表示外部已存在的库

- 不参与编译

- 只有属性

### Source/Property 层级

```
target_sources(my_lib PRIVATE a.cpp b.cpp)
```

- source 属于 target

- source 本身不是独立层级

总结来说，Cmake就在做三件事：

1. **定义 target**

2. **链接 target**

3. **给 target 设置属性**