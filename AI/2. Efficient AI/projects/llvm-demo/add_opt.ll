; ModuleID = 'add.ll'
source_filename = "add.ll"

define i32 @add(i32 %a, i32 %b) {
entry:
  %c = add i32 %a, %b
  ret i32 %c
}
