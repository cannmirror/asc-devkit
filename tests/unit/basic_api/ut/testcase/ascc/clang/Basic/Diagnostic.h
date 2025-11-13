#ifndef CLANG_BASIC_DIAGNOSTIC_H
#define CLANG_BASIC_DIAGNOSTIC_H
#include <memory>
#include "clang/Basic/DiagnosticIDs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
class Diagnostic {
public:
    Diagnostic() = default;
    void FormatDiagnostic(llvm::SmallVectorImpl<char> &OutStr) const { return; }
    SourceManager getSourceManager() const { return SourceManager(); }
    SourceLocation getLocation() const { return SourceLocation(); }
public:
};

class DiagnosticConsumer;

class DiagnosticsEngine {
private:
    bool SuppressAllDiagnostics = false;
    class DiagState {
        llvm::DenseMap<unsigned, DiagnosticMapping> DiagMap;
    public:
        unsigned IgnoreAllWarnings : 1;
    };
public:
    explicit DiagnosticsEngine() {};
    DiagnosticsEngine(const DiagnosticsEngine &) = delete;
    DiagnosticsEngine &operator=(const DiagnosticsEngine &) = delete;
    ~DiagnosticsEngine() {};

    class DiagStateMap {
    private:
        std::shared_ptr<DiagState> CurDiagState;
    public:
        std::shared_ptr<DiagState> getCurDiagState()
        {
            CurDiagState = std::make_shared<DiagState>();
            return CurDiagState;
        }
    };
    DiagStateMap DiagStatesByLoc;

    enum Level {
        Ignored = DiagnosticIDs::Ignored,
        Note = DiagnosticIDs::Note,
        Remark = DiagnosticIDs::Remark,
        Warning = DiagnosticIDs::Warning,
        Error = DiagnosticIDs::Error,
        Fatal = DiagnosticIDs::Fatal
    };

    void setSuppressAllDiagnostics(bool Val = true) {
        SuppressAllDiagnostics = Val;
    }

    DiagState *GetCurDiagState() {
        return DiagStatesByLoc.getCurDiagState().get();
    }

    void setIgnoreAllWarnings(bool Val) {
        GetCurDiagState()->IgnoreAllWarnings = Val;
    }

    void setClient(DiagnosticConsumer *client, bool ShouldOwnClient)
    {
        std::unique_ptr<DiagnosticConsumer> consumerPtr(client);
    }

    void setErrorLimit(unsigned Limit) {}
};

class DiagnosticConsumer {
public:
    DiagnosticConsumer() = default;

    virtual void HandleDiagnostic(clang::DiagnosticsEngine::Level DiagLevel,
                                const clang::Diagnostic &Info) {}

    virtual bool IncludeInDiagnosticCounts() const {}
};
}
#endif