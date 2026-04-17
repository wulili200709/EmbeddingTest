#pragma once
#include<Windows.h>
#include <qthread.h>
#include "NKLCLIB.h"
#include "NKIOLIB.h"

typedef int(*NKLC_LibraryInit_t)(const char *configFile);
typedef int(*NKLC_OpenDevice_t)(unsigned short port, pLcCallbackFunc pCallBackFun);
typedef int(*NKLC_CloseDevice_t)(unsigned short port, pLcCallbackFunc pCallBackFun);
typedef int(*NKLC_IsDeviceOpened_t)(unsigned short port, unsigned int devId);
typedef int(*NKLC_Process_t)(void);
typedef int(*NKLC_LibraryDeinit_t)();


typedef int(*NKLC_GetVerInfo_t)(unsigned int devId, pLcCallbackFunc pCallBackFun);

typedef int(*NKLC_SetPwmParams_t)(unsigned int devId,
	unsigned char ucChIdx,
	unsigned char ucPwmMode,
	unsigned char ucPwmValue,
	unsigned char ucPwmHoldingTime,
	unsigned char ucPwmOnOff,
	pLcCallbackFunc pCallBackFun);

typedef int(*NKLC_GetPwmParams_t)(unsigned int devId,
	unsigned char ucChIdx,
	pLcCallbackFunc pCallBackFun);

typedef int(*NKLC_IAPDownload_t)(unsigned int devId,
	const char *imageFile,
	pLcCallbackFunc pCallBackIAPDownload);

typedef int(*NKLC_SetGeneralParam_t)(unsigned int devId,
	unsigned char ucParamId,
	unsigned char ucParamLen,
	unsigned int ucParamValue,
	pLcCallbackFunc pCallBackFun);

typedef int(*NKLC_GetGeneralParam_t)(unsigned int devId,
	unsigned char ucParamId,
	unsigned char ucParamLen,
	pLcCallbackFunc pCallBackFun);


// DIO Functions
typedef int(*NKDIO_LibraryInit_t)(const char *configFile);
typedef int(*NKDIO_PollingReadDiByte_t)(unsigned char diByteIndex, unsigned char *pByteValue);
typedef int(*NKDIO_PollingReadDiWord_t)(unsigned char diWordIndex, unsigned short *pWordValue);
typedef int(*NKDIO_PollingWriteDoByte_t)(unsigned char doByteIndex, unsigned char doByteValue);
typedef int(*NKDIO_PollingWriteDoWord_t)(unsigned char doWordIndex, unsigned short doWordValue);
typedef void(*NKDIO_LibraryDeinit_t)(void);
// Read and Write DIO in polling mode
/*-Polling Mode API------------------------------------------------------------------------------*/






class CDioLcAdapter :
	public QThread
{
	Q_OBJECT
public:
	CDioLcAdapter(QObject *parent = nullptr);
	~CDioLcAdapter();
	static CDioLcAdapter *pThis;



public slots:

	int slotBoardOpenPort(unsigned int devId, unsigned short port);
	int slotBoardClosePort(unsigned int devId, unsigned short port);
	int slotBoardIsDeviceOpened(unsigned char devId);

	int slotBoardGetVerInfo(unsigned int devId);

	int slotBoardSetPwmParams(unsigned int devId,
		unsigned char ucChIdx,
		unsigned char ucPwmMode,
		unsigned char ucPwmValue,
		unsigned char ucPwmHoldingTime,
		unsigned char ucPwmOnOff);

	int slotBoardGetPwmParams(unsigned int devId, unsigned char ucChIdx);

	int slotBoardIAPDownload(unsigned int devId, QString imageFile);


	// for the general params settings
	int slotSetGeneralParam(unsigned int devId, unsigned char paramId, unsigned char ucParamLen, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId);
	int slotGetGeneralParam(unsigned int devId, unsigned char paramId, unsigned char ucParmaLen, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId);


signals:
	void signalComOpenCB(unsigned short portNum,
		unsigned char  ucHardwareMajorVer,
		unsigned char  ucHardwareMinorVer,
		unsigned char  ucHardwareRevVer,
		unsigned char  ucFirmwareMajorVer,
		unsigned char  ucFirmwareMinorVer,
		unsigned char  ucFirmwareRevVer,
		unsigned char ucError,
		unsigned int uiErrorId);
	void signalComCloseCB(unsigned char ucError, unsigned int uiErrorId);
	void signalGetDeviceVerCB(
		unsigned char  ucHardwareMajorVer,
		unsigned char  ucHardwareMinorVer,
		unsigned char  ucHardwareRevVer,
		unsigned char  ucFirmwareMajorVer,
		unsigned char  ucFirmwareMinorVer,
		unsigned char  ucFirmwareRevVer,
		unsigned char  ucError,
		unsigned int  uiErrorId);
	void signalSetPwmParamsCB(unsigned char  ucChIdx, unsigned char ucError, unsigned int uiErrorId);
	void signalGetPwmParamsCB(unsigned char  ucChIdx,
		unsigned char  ucPwmMode,
		unsigned char  ucPwmValue,
		unsigned char  ucPwmHoldingTime,
		unsigned char  ucPwmOnOff,
		unsigned char ucError,
		unsigned int uiErrorId);
	void signalIAPDownloadCB(unsigned char ucDevId, unsigned int uiFileTotalSize, unsigned int uiFileSizeDownloaded, unsigned char ucError, unsigned int uiErrorId);


	void signalSetGeneralParamCB(unsigned char paramId, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId);
	void signalGetGeneralParamCB(unsigned char paramId, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId);

public:
	NKLC_LibraryInit_t NKLC_LibraryInit;
	NKLC_LibraryDeinit_t   NKLC_LibraryDeinit;
	NKLC_Process_t NKLC_Process;
	NKLC_OpenDevice_t   NKLC_OpenDevice;
	NKLC_CloseDevice_t  NKLC_CloseDevice;
	NKLC_IsDeviceOpened_t NKLC_IsDeviceOpened;

	NKLC_GetVerInfo_t NKLC_GetVerInfo;
	NKLC_SetPwmParams_t NKLC_SetPwmParams;
	NKLC_GetPwmParams_t NKLC_GetPwmParams;
	NKLC_IAPDownload_t NKLC_IAPDownload;
	NKLC_GetGeneralParam_t NKLC_GetGeneralParam;
	NKLC_SetGeneralParam_t NKLC_SetGeneralParam;

	HMODULE          hDIO;
	NKDIO_LibraryInit_t				 NKDIO_LibraryInit;
	NKDIO_PollingReadDiByte_t		 NKDIO_PollingReadDiByte;
	NKDIO_PollingReadDiWord_t		 NKDIO_PollingReadDiWord;
	NKDIO_PollingWriteDoByte_t		 NKDIO_PollingWriteDoByte;
	NKDIO_PollingWriteDoWord_t		 NKDIO_PollingWriteDoWord;
	NKDIO_LibraryDeinit_t			 NKDIO_LibraryDeinit;

	unsigned short m_bDioNum;
	bool m_bLcEnabled;
	bool m_bAioEnabled;
	bool m_bShowAboutPage;
	unsigned short m_port;

	// DIO functions
	/*-Polling Mode API------------------------------------------------------------------------------*/

protected:
	void run();

private:

	int pvInit();
	int IOInit();

	// Light control CALL backs
	static void ComOpenCB(LC_CALLBACK_ARG_T Arg);
	static void ComCloseCB(LC_CALLBACK_ARG_T Arg);
	static void GetDeviceVerCB(LC_CALLBACK_ARG_T Arg);
	static void SetPwmParamsCB(LC_CALLBACK_ARG_T Arg);
	static void GetPwmParamsCB(LC_CALLBACK_ARG_T Arg);

	static void SetGeneralParamCB(LC_CALLBACK_ARG_T Arg);
	static void GetGeneralParamCB(LC_CALLBACK_ARG_T Arg);

	static void IAPDownloadCB(LC_CALLBACK_ARG_T pArg);


	HMODULE          hLC;
	//HMODULE          hDIO;

	int             m_bLoaded;

};

