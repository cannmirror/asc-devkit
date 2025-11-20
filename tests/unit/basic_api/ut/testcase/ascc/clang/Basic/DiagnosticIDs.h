#ifndef CLANG_BASIC_DIAGNOSTICIDS_H
#define CLANG_BASIC_DIAGNOSTICIDS_H

namespace clang {
class DiagnosticIDs {
public:
    enum Level {
        Ignored, Note, Remark, Warning, Error, Fatal
    };
};

class DiagnosticMapping {
    unsigned Severity : 3;
    unsigned IsUser : 1;
    unsigned IsPragma : 1;
    unsigned HasNoWarningAsError : 1;
    unsigned HasNoErrorAsFatal : 1;
    unsigned WasUpgradedFromWarning : 1;
public:
    DiagnosticMapping() = default;
};
}
#endif