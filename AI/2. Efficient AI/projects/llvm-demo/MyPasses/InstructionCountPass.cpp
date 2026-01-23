#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

struct InstructionCountPass : public PassInfoMixin<InstructionCountPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
        int count = 0;
        for (auto &BB : F)
            for (auto &I : BB)
                count++;
        errs() << "Function " << F.getName() << " has " << count << " instructions\n";
        return PreservedAnalyses::all();
    }
};

// 注册 pass
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "InstructionCountPass", "v0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "inst-count") {
                        FPM.addPass(InstructionCountPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
