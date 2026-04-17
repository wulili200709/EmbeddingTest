#ifndef _NKLCLIB_H_
#define _NKLCLIB_H_
#pragma once
#include <Windows.h>


#ifdef NK_EXPORTS
#define NK_API __declspec(dllexport)
#else
#define NK_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif
	typedef enum
	{
		E_NOERR = 0,              /*!< No error. */
		E_NOREG = 1,              /*!< Illegal register address. */
		E_INVAL = 2,              /*!< Illegal argument. */
		E_PORTERR = 3,            /*!< Porting layer error. */
		E_NORES = 4,              /*!< Insufficient resources. */
		E_IO = 5,                 /*!< I/O error. */
		E_TIMEDOUT = 6           /*!< Timeout error occurred. */
	} eErrorCode;

	typedef struct _OPENCOM_CALLBACK_ARG_T
	{
		unsigned short portNum;
		unsigned char devId;
		unsigned char hardwareMajorVer;
		unsigned char hardwareMinorVer;
		unsigned char hardwareRevVer;
		unsigned char firmwareMajorVer;
		unsigned char firmwareMinorVer;
		unsigned char firmwareRevVer;
		unsigned char fillup_1;
		unsigned char fillup_2;
		unsigned char error;
		unsigned int errorId;

	}OPENCOM_CALLBACK_ARG_T;

	typedef struct _CLOSECOM_CALLBACK_ARG_T
	{
		unsigned int fill_up;
		unsigned char fill_up_1;
		unsigned char fill_up_2;
		unsigned char fill_up_3;
		unsigned char error;
		unsigned int errorId;
		unsigned int fill_up_4;
	}CLOSECOM_CALLBACK_ARG_T;

	typedef struct _GET_DEVICE_VER_CALLBACK_ARG_T
	{
		unsigned char hardwareMajorVer;
		unsigned char hardwareMinorVer;
		unsigned char hardwareRevVer;
		unsigned char firmwareMajorVer;
		unsigned char firmwareMinorVer;
		unsigned char firmwareRevVer;
		unsigned char fill_up;
		unsigned char error;
		unsigned int errorId;
		unsigned int fill_up_2;
	}GET_DEVICE_VER_CALLBACK_ARG_T;

	typedef struct _SET_PWM_PARAMS_CALLBACK_ARG_T
	{
		unsigned int fill_up;
		unsigned char chIdx;
		unsigned char fill_up_1;
		unsigned char fill_up_2;
		unsigned char error;
		unsigned int errorId;
		unsigned int fill_up_3;
	}SET_PWM_PARAMS_CALLBACK_ARG_T;

	typedef struct _GET_PWM_PARAMS_CALLBACK_ARG_T
	{
		unsigned char chIdx;
		unsigned char pwmMode;
		unsigned char pwmValue;
		unsigned char pwmHoldingTime;
		unsigned char pwmOnOff;
		unsigned char fill_up_2;
		unsigned char fill_up_3;
		unsigned char error;
		unsigned int errorId;
		unsigned int fill_up_4;
	}GET_PWM_PARAMS_CALLBACK_ARG_T;

	typedef struct _SET_GENERAL_PARAM_CALLBACK_ARG_T
	{
		unsigned char devId;
		unsigned char cmdId;
		unsigned char paramId;
		unsigned char paramLen;
		unsigned int  paramValue;
		unsigned char fill_up_1;
		unsigned char fill_up_2;
		unsigned char fill_up_3;
		unsigned char error;
		unsigned int errorId;
	}SET_GENERAL_PARAM_CALLBACK_ARG_T;

	typedef struct _GET_GENERAL_PARAM_CALLBACK_ARG_T
	{
		unsigned char devId;
		unsigned char cmdId;
		unsigned char paramId;
		unsigned char paramLen;
		unsigned int  paramValue;
		unsigned char fill_up_1;
		unsigned char fill_up_2;
		unsigned char fill_up_3;
		unsigned char error;
		unsigned int errorId;
	}GET_GENERAL_PARAM_CALLBACK_ARG_T;


	typedef struct _IAP_CALLBACK_ARG_T
	{
		unsigned char devId;
		unsigned char error;
		unsigned char fill_up_1;
		unsigned char fill_up_2;
		unsigned int errorId;
		unsigned int totalFileSize;
		unsigned int downloadedSize;

	}IAP_CALLBACK_ARG_T;

	// for the older version 
	
	typedef union
	{
		OPENCOM_CALLBACK_ARG_T openComCallbackArg;
		CLOSECOM_CALLBACK_ARG_T closeComCallbackArg;
		GET_DEVICE_VER_CALLBACK_ARG_T getDeviceVerCallbackArg;
		SET_PWM_PARAMS_CALLBACK_ARG_T setPwmParamsCallbackArg;
		GET_PWM_PARAMS_CALLBACK_ARG_T getPwmParamsCallbackArg;
		SET_GENERAL_PARAM_CALLBACK_ARG_T setGeneralParamCallbackArg;
		GET_GENERAL_PARAM_CALLBACK_ARG_T getGeneralParamCallbackArg;
		IAP_CALLBACK_ARG_T iapUpdateCallbackArg;
	}LC_CALLBACK_ARG_T;

	typedef void(*pLcCallbackFunc)(LC_CALLBACK_ARG_T arg);


	NK_API int NKLC_LibraryInit(const char *configFile);
	NK_API int NKLC_LibraryDeinit();


	/*************************************************************************************************/
	/*                                                                                               */
	/*    Nodka Light Control library	in Async  Mode												 */
	/*                                                                                               */
	/*************************************************************************************************/
	NK_API int  NKLC_OpenDevice_Async(unsigned short port, pLcCallbackFunc pCallBackFun);
	NK_API int  NKLC_CloseDevice_Async(unsigned short port, pLcCallbackFunc pCallBackFun);
	NK_API int  NKLC_IsDeviceOpened_Async(unsigned short port, unsigned int devId);
	NK_API int  NKLC_Process_Async(void);
	

	NK_API int NKLC_GetVerInfo_Async(unsigned int devId, pLcCallbackFunc pCallBackFun);

	NK_API int NKLC_SetPwmParams_Async(unsigned int devId,
		unsigned char ucChIdx,
		unsigned char ucPwmMode,
		unsigned char ucPwmValue,
		unsigned char ucPwmHoldingTime,
		unsigned char ucPwmOnOff,
		pLcCallbackFunc pCallBackFun);

	NK_API int NKLC_GetPwmParams_Async(unsigned int devId,
		unsigned char ucChIdx,
		pLcCallbackFunc pCallBackFun);



	NK_API int NKLC_IAPDownload_Async(unsigned int devId,
		const char *imageFile,
		pLcCallbackFunc pCallBackIAPDownload);



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Add for the parameters settings in Async  Mode	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	NK_API int NKLC_SetGeneralParam_Async(unsigned int devId,
		unsigned char ucParamId,
		unsigned char ucParamLen,
		unsigned int uiParamValue,
		pLcCallbackFunc pCallBackFun);

	NK_API int NKLC_GetGeneralParam_Async(unsigned int devId,
		unsigned char ucParamId,
		unsigned char ucParamLen,
		pLcCallbackFunc pCallBackFun);




	/*************************************************************************************************/
	/*                                                                                               */
	/*    Nodka Light Control library																 */
	/*                                                                                               */
	/*************************************************************************************************/
	NK_API int  NKLC_OpenDevice(unsigned short port, pLcCallbackFunc pCallBackFun);
	NK_API int  NKLC_CloseDevice(unsigned short port, pLcCallbackFunc pCallBackFun);
	NK_API int  NKLC_IsDeviceOpened(unsigned short port, unsigned int devId);
	NK_API int  NKLC_GetVerInfo(unsigned int devId, pLcCallbackFunc pCallBackFun);
	NK_API int  NKLC_SetPwmParams(unsigned int devId,
		unsigned char ucChIdx,
		unsigned char ucPwmMode,
		unsigned char ucPwmValue,
		unsigned char ucPwmHoldingTime,
		unsigned char ucPwmOnOff,
		pLcCallbackFunc pCallBackFun);
	NK_API int NKLC_GetPwmParams(unsigned int devId,
		unsigned char ucChIdx,
		pLcCallbackFunc pCallBackFun);

	NK_API int NKLC_IAPDownload(unsigned int devId,
		const char *imageFile,
		pLcCallbackFunc pCallBackIAPDownload);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Add for the parameters settings 	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	NK_API int NKLC_SetGeneralParam(unsigned int devId,
		unsigned char ucParamId,
		unsigned char ucParamLen,
		unsigned int uiParamValue,
		pLcCallbackFunc pCallBackFun);

	NK_API int NKLC_GetGeneralParam(unsigned int devId,
		unsigned char ucParamId,
		unsigned char ucParamLen,
		pLcCallbackFunc pCallBackFun);

#ifdef __cplusplus
}
#endif


#endif // _NK_LC_LIB_H_