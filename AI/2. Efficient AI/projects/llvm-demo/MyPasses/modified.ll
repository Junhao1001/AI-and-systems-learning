; ModuleID = 'test.ll'
source_filename = "test.c"

define i32 @main() {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 10, ptr %a, align 4
  store i32 20, ptr %b, align 4
  %x = load i32, ptr %a, align 4
  %y = load i32, ptr %b, align 4
  %0 = sub i32 %x, %y
  ret i32 %0
}
