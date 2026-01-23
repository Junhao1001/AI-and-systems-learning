; ModuleID = 'test'
source_filename = "test.c"

define i32 @main() {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 10, i32* %a, align 4
  store i32 20, i32* %b, align 4
  %x = load i32, i32* %a, align 4
  %y = load i32, i32* %b, align 4
  %sum = add i32 %x, %y
  ret i32 %sum
}
