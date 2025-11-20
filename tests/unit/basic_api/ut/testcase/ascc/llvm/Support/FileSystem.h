#ifndef LLVM_SUPPORT_FILESYSTEM_H
#define LLVM_SUPPORT_FILESYSTEM_H

namespace llvm {
namespace sys {
namespace fs {
enum OpenFlags : unsigned {
    OF_None = 0,
    F_None = 0, // For compatibility

    /// The file should be opened in text mode on platforms that make this
    /// distinction.
    OF_Text = 1,
    F_Text = 1, // For compatibility

    /// The file should be opened in append mode.
    OF_Append = 2,
    F_Append = 2, // For compatibility

    /// Delete the file on close. Only makes a difference on windows.
    OF_Delete = 4,

    /// When a child process is launched, this file should remain open in the
    /// child process.
    OF_ChildInherit = 8,

    /// Force files Atime to be updated on access. Only makes a difference on windows.
    OF_UpdateAtime = 16,
};
} // namespace fs
} // namespace sys
} // namespace llvm

#endif // LLVM_SUPPORT_FILESYSTEM_H