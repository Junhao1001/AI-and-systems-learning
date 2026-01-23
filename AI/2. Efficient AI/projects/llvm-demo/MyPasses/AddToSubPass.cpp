#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

struct AddToSubPass : public PassInfoMixin<AddToSubPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
        SmallVector<BinaryOperator*, 8> Adds;

    // 第一步：只遍历，不修改
        for (auto &BB : F) {
            for (auto &I : BB) {
                if (auto *op = dyn_cast<BinaryOperator>(&I)) {
                    if (op->getOpcode() == Instruction::Add) {
                        Adds.push_back(op);
                    }
                }
            }
        }

    // 第二步：安全地修改
        for (auto *op : Adds) {
            IRBuilder<> builder(op);
            auto *newSub = builder.CreateSub(
                op->getOperand(0), op->getOperand(1)
            );
            op->replaceAllUsesWith(newSub);
            op->eraseFromParent();
        }
        
        return PreservedAnalyses::none();
    }
};

// 注册 pass
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "AddtoSubPass", "v0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "add2sub") {
                        FPM.addPass(AddToSubPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
