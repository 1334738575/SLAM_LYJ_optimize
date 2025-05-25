#ifndef OPTIMIZE_LYJ_DEFINES_H
#define OPTIMIZE_LYJ_DEFINES_H

// export
#ifdef WIN32
#ifdef _MSC_VER
#define OPTIMIZE_LYJ_API __declspec(dllexport)
#else
#define OPTIMIZE_LYJ_API
#endif
#else
#define OPTIMIZE_LYJ_API
#endif

namespace OPTIMIZE_LYJ
{

}

#endif