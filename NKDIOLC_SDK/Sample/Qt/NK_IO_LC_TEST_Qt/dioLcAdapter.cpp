#include "qmessagebox.h"
#include "qcoreapplication.h"
#include "qsettings.h"
#include "DioLcAdapter.h"

CDioLcAdapter *CDioLcAdapter::pThis = NULL;

CDioLcAdapter::CDioLcAdapter(QObject *parent) :QThread(parent)
{
	m_bLoaded = -1;
	pThis = this;
	m_bLcEnabled = false;
	m_bAioEnabled = false;
	int ret = 0;
	ret = IOInit();
	if (0 != ret)
	{
		QMessageBox::warning(0, tr("Warning"),
			tr("Load DIO library Failed!") + QString("ret:%1").arg(ret),
			QMessageBox::Ok);
		m_bDioNum = 0;
	}
	ret = pvInit();
	if (0 != ret)
	{
		QMessageBox::warning(0, tr("Warning"),
			tr("Load LightControl library Failed!") + QString("ret:%1").arg(ret),
			QMessageBox::Ok);
		return;
	}



}
CDioLcAdapter::~CDioLcAdapter()
{

}

// slot to open the device port, for the H1, the default serial port is 3
// return the version information in the callback if connected to the MCU successfully
int CDioLcAdapter::slotBoardOpenPort(unsigned int devId, unsigned short port)
{
	return NKLC_OpenDevice(port, this->ComOpenCB);
}
// slot to close the device port
int CDioLcAdapter::slotBoardClosePort(unsigned int devId, unsigned short port)
{
	return NKLC_CloseDevice(port, this->ComCloseCB);
}
// slot to check if the port has been opened or not, return 1 when has been opened, otherwize return 0;
int CDioLcAdapter::slotBoardIsDeviceOpened(unsigned char devId)
{
	return NKLC_IsDeviceOpened(3, devId);
}

// slot to read the hardware version and firmware information 
int CDioLcAdapter::slotBoardGetVerInfo(unsigned int devId)
{
	return NKLC_GetVerInfo(devId, this->GetDeviceVerCB);
}

// slot to set the light control parameters to the specified channel
int CDioLcAdapter::slotBoardSetPwmParams(unsigned int devId,
	unsigned char ucChIdx,
	unsigned char ucPwmMode,
	unsigned char ucPwmValue,
	unsigned char ucPwmHoldingTime,
	unsigned char ucPwmOnOff)
{
	return NKLC_SetPwmParams(devId, ucChIdx, ucPwmMode, ucPwmValue, ucPwmHoldingTime, ucPwmOnOff, SetPwmParamsCB);
}

// slot to read the light control parameters from the according channel.
int CDioLcAdapter::slotBoardGetPwmParams(unsigned int devId, unsigned char ucChIdx)
{
	return NKLC_GetPwmParams(devId, ucChIdx, GetPwmParamsCB);
}

int CDioLcAdapter::slotBoardIAPDownload(unsigned int devId, QString imageFile)
{
	std::string str = imageFile.toStdString();
	QByteArray ba = imageFile.toLocal8Bit();
	char *ch = ba.data();

	return NKLC_IAPDownload(devId, ch, IAPDownloadCB);
}
// for the general params settings
int CDioLcAdapter::slotSetGeneralParam(unsigned int devId, unsigned char paramId, unsigned char ucParamLen, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId)
{
	return NKLC_SetGeneralParam(devId, paramId, ucParamLen, paramValue, SetGeneralParamCB);
}
int CDioLcAdapter::slotGetGeneralParam(unsigned int devId, unsigned char paramId, unsigned char ucParmaLen, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId)
{
	return NKLC_GetGeneralParam(devId, paramId, ucParmaLen, GetGeneralParamCB);
}
// the cycle thread to do the process work
void CDioLcAdapter::run()
{
#if 1
	while (1)
	{
		if (m_bLcEnabled)
		{
			NKLC_Process();
		}
		QThread::msleep(1);
	}

#endif
}


// load dynamic library and initialized the API address
int CDioLcAdapter::pvInit()
{
#if 1
	int ret = -1;
	//#ifdef Q_OS_WIN64
#ifdef WIN64
	hLC = GetModuleHandle(TEXT("./NKLCLIBx64.dll"));
	if (hLC == NULL)
	{
		hLC = LoadLibrary(TEXT("./NKLCLIBx64.dll"));
	}
#else ifdef WIN32
	hLC = GetModuleHandle(TEXT("./NKLCLIBx86.dll"));
	if (hLC == NULL)
	{
		hLC = LoadLibrary(TEXT("./NKLCLIBx86.dll"));
	}
#endif


	if (hLC == NULL)
		return -1;

	NKLC_LibraryInit = (NKLC_LibraryInit_t)GetProcAddress(hLC, "NKLC_LibraryInit");
	NKLC_LibraryDeinit = (NKLC_LibraryDeinit_t)GetProcAddress(hLC, "NKLC_LibraryDeinit");
	NKLC_Process = (NKLC_Process_t)GetProcAddress(hLC, "NKLC_Process_Async");
#if 1
	NKLC_OpenDevice = (NKLC_OpenDevice_t)GetProcAddress(hLC, "NKLC_OpenDevice_Async");
	NKLC_CloseDevice = (NKLC_CloseDevice_t)GetProcAddress(hLC, "NKLC_CloseDevice_Async");
	NKLC_IsDeviceOpened = (NKLC_IsDeviceOpened_t)GetProcAddress(hLC, "NKLC_IsDeviceOpened_Async");
	NKLC_GetVerInfo = (NKLC_GetVerInfo_t)GetProcAddress(hLC, "NKLC_GetVerInfo_Async");
	NKLC_SetPwmParams = (NKLC_SetPwmParams_t)GetProcAddress(hLC, "NKLC_SetPwmParams_Async");
	NKLC_GetPwmParams = (NKLC_GetPwmParams_t)GetProcAddress(hLC, "NKLC_GetPwmParams_Async");
	NKLC_IAPDownload = (NKLC_IAPDownload_t)GetProcAddress(hLC, "NKLC_IAPDownload_Async");
	NKLC_GetGeneralParam = (NKLC_GetGeneralParam_t)GetProcAddress(hLC, "NKLC_GetGeneralParam_Async");
	NKLC_SetGeneralParam = (NKLC_SetGeneralParam_t)GetProcAddress(hLC, "NKLC_SetGeneralParam_Async");
#endif

#if 0
	NKLC_OpenDevice = (NKLC_OpenDevice_t)GetProcAddress(hLC, "NKLC_OpenDevice");
	NKLC_CloseDevice = (NKLC_CloseDevice_t)GetProcAddress(hLC, "NKLC_CloseDevice");
	NKLC_IsDeviceOpened = (NKLC_IsDeviceOpened_t)GetProcAddress(hLC, "NKLC_IsDeviceOpened");
	NKLC_GetVerInfo = (NKLC_GetVerInfo_t)GetProcAddress(hLC, "NKLC_GetVerInfo");
	NKLC_SetPwmParams = (NKLC_SetPwmParams_t)GetProcAddress(hLC, "NKLC_SetPwmParams");
	NKLC_GetPwmParams = (NKLC_GetPwmParams_t)GetProcAddress(hLC, "NKLC_GetPwmParams");
	NKLC_IAPDownload = (NKLC_IAPDownload_t)GetProcAddress(hLC, "NKLC_IAPDownload");
	NKLC_GetGeneralParam = (NKLC_GetGeneralParam_t)GetProcAddress(hLC, "NKLC_GetGeneralParam");
	NKLC_SetGeneralParam = (NKLC_SetGeneralParam_t)GetProcAddress(hLC, "NKLC_SetGeneralParam");
#endif
	// DIO functions
	 //DIO_Init = (DIO_Init_t)GetProcAddress(hDIO,"DIO_Init");
	/*-Polling Mode API------------------------------------------------------------------------------*/

	

	if ((NULL != NKLC_LibraryInit)
		&& (NULL != NKLC_LibraryDeinit)
		&& (NULL != NKLC_Process)
		&& (NULL != NKLC_OpenDevice)
		&& (NULL != NKLC_CloseDevice)
		&& (NULL != NKLC_IsDeviceOpened)
		&& (NULL != NKLC_GetVerInfo)
		&& (NULL != NKLC_SetPwmParams)
		&& (NULL != NKLC_GetPwmParams)
		&& (NULL != NKLC_GetGeneralParam)
		&& (NULL != NKLC_SetGeneralParam))
	{
		//QString qexeFullPath = QCoreApplication::applicationDirPath();
		//QString filePath = qexeFullPath + "/nkio_config.ini";
		QSettings set(QCoreApplication::applicationDirPath() + "/select.ini", QSettings::IniFormat);
		QString filePath = QCoreApplication::applicationDirPath() + set.value("/SELECTED/ConfigPath").toString();
		QSettings configSet(filePath, QSettings::IniFormat);
		m_bLcEnabled = configSet.value("/NKLC/Enabled").toBool();
		m_port = configSet.value("/NKLC/PortNum").toUInt();
		filePath.replace(QString("/"), QString("\\"));
		QByteArray ba = filePath.toLocal8Bit();
		char *ch = ba.data();
		//workThread->init(ch);
		//if (m_bLcEnabled)
		//{
			int lc_ret = NKLC_LibraryInit(ch);
			if (lc_ret == 0)
			{

				ret = lc_ret;
			}
			else
			{
				ret =  lc_ret;
			}
		//}
		//else
		//{
		//	ret = -1;
		//}

	}
	m_bLoaded = ret;
	if (ret != 0)
	{
		FreeLibrary(hLC);
		hLC = NULL;
	}

	return ret;
#endif

}


int CDioLcAdapter::IOInit()
{
	int ret = false;
	//#ifdef Q_OS_WIN64
#ifdef WIN64
	hDIO = GetModuleHandle(TEXT("./NKIOLIBx64.dll"));
	if (hDIO == NULL)
	{
		hDIO = LoadLibrary(TEXT("./NKIOLIBx64.dll"));
	}
#else ifdef WIN32
	hDIO = GetModuleHandle(TEXT("./NKIOLIBx86.dll"));
	if (hDIO == NULL)
	{
		hDIO = LoadLibrary(TEXT("./NKIOLIBx86.dll"));
	}
#endif


	if (hDIO == NULL)
		return false;

	NKDIO_LibraryInit = (NKDIO_LibraryInit_t)GetProcAddress(hDIO, "NKDIO_LibraryInit");
	NKDIO_LibraryDeinit = (NKDIO_LibraryDeinit_t)GetProcAddress(hDIO, "NKDIO_LibraryDeinit");
	NKDIO_PollingReadDiByte = (NKDIO_PollingReadDiByte_t)GetProcAddress(hDIO, "NKDIO_PollingReadDiByte");
	NKDIO_PollingReadDiWord = (NKDIO_PollingReadDiWord_t)GetProcAddress(hDIO, "NKDIO_PollingReadDiWord");
	NKDIO_PollingWriteDoByte = (NKDIO_PollingWriteDoByte_t)GetProcAddress(hDIO, "NKDIO_PollingWriteDoByte");
	NKDIO_PollingWriteDoWord = (NKDIO_PollingWriteDoWord_t)GetProcAddress(hDIO, "NKDIO_PollingWriteDoWord");



	if ((NULL != NKDIO_LibraryInit)
		&& (NULL != NKDIO_LibraryDeinit)
		&& (NULL != NKDIO_PollingReadDiByte)
		&& (NULL != NKDIO_PollingReadDiWord)
		&& (NULL != NKDIO_PollingWriteDoByte)
		&& (NULL != NKDIO_PollingWriteDoWord))
	{
		QSettings set(QCoreApplication::applicationDirPath() + "/select.ini", QSettings::IniFormat);
		QString filePath = QCoreApplication::applicationDirPath() + set.value("/SELECTED/ConfigPath").toString();
		QSettings configSet(filePath, QSettings::IniFormat);
		m_bDioNum = configSet.value("/NKDIO/DeviceNum").toUInt();
		filePath.replace(QString("/"), QString("\\"));
		QByteArray ba = filePath.toLocal8Bit();
		char *ch = ba.data();

		if (m_bDioNum >= 1)
		{
			int dio_ret = NKDIO_LibraryInit(ch);
			qDebug("dio_ret:%d", dio_ret);
			if (dio_ret == 0)
			{
				ret = 0;
			}
			else
			{
				ret = dio_ret;
			}
		}
		else
		{
			ret = -1;
		}


	}
	m_bLoaded = ret;
	if (ret != 0)
	{
		FreeLibrary(hDIO);
		hDIO = NULL;
	}
	return ret;
}


// called when the port opened successfully
void CDioLcAdapter::ComOpenCB(LC_CALLBACK_ARG_T Arg)
{
	emit pThis->signalComOpenCB(Arg.openComCallbackArg.portNum, 
		Arg.openComCallbackArg.hardwareMajorVer,
		Arg.openComCallbackArg.hardwareMinorVer,
		Arg.openComCallbackArg.hardwareRevVer,
		Arg.openComCallbackArg.firmwareMajorVer,
		Arg.openComCallbackArg.firmwareMinorVer,
		Arg.openComCallbackArg.firmwareRevVer,
		Arg.openComCallbackArg.error, 
		Arg.openComCallbackArg.errorId);
}
// called when the port is closed 
void CDioLcAdapter::ComCloseCB(LC_CALLBACK_ARG_T Arg)
{
	emit pThis->signalComCloseCB(Arg.closeComCallbackArg.error, Arg.closeComCallbackArg.errorId);
}

// called when got the version information successfully
void CDioLcAdapter::GetDeviceVerCB(LC_CALLBACK_ARG_T Arg)
{
	emit pThis->signalGetDeviceVerCB(Arg.getDeviceVerCallbackArg.hardwareMajorVer,
		Arg.getDeviceVerCallbackArg.hardwareMinorVer,
		Arg.getDeviceVerCallbackArg.hardwareRevVer,
		Arg.getDeviceVerCallbackArg.firmwareMajorVer,
		Arg.getDeviceVerCallbackArg.firmwareMinorVer,
		Arg.getDeviceVerCallbackArg.firmwareRevVer,
		Arg.getDeviceVerCallbackArg.error,
		Arg.getDeviceVerCallbackArg.errorId);
}

// called when set the light control pararmters successfully
void CDioLcAdapter::SetPwmParamsCB(LC_CALLBACK_ARG_T Arg)
{
	emit pThis->signalSetPwmParamsCB(Arg.setPwmParamsCallbackArg.chIdx, Arg.setPwmParamsCallbackArg.error, Arg.setPwmParamsCallbackArg.errorId);
}

// called when read the light control parameters successfully
void CDioLcAdapter::GetPwmParamsCB(LC_CALLBACK_ARG_T Arg)
{
	emit pThis->signalGetPwmParamsCB(Arg.getPwmParamsCallbackArg.chIdx,
		Arg.getPwmParamsCallbackArg.pwmMode,
		Arg.getPwmParamsCallbackArg.pwmValue,
		Arg.getPwmParamsCallbackArg.pwmHoldingTime,
		Arg.getPwmParamsCallbackArg.pwmOnOff,
		Arg.getPwmParamsCallbackArg.error,
		Arg.getPwmParamsCallbackArg.errorId);
}

void CDioLcAdapter::IAPDownloadCB(LC_CALLBACK_ARG_T pArg)
{
	emit pThis->signalIAPDownloadCB(pArg.iapUpdateCallbackArg.devId, pArg.iapUpdateCallbackArg.totalFileSize,pArg.iapUpdateCallbackArg.downloadedSize,pArg.iapUpdateCallbackArg.error, pArg.iapUpdateCallbackArg.errorId);
}

void CDioLcAdapter::SetGeneralParamCB(LC_CALLBACK_ARG_T Arg)
{
	emit pThis->signalSetGeneralParamCB(Arg.setGeneralParamCallbackArg.paramId,
		Arg.setGeneralParamCallbackArg.paramValue,
		Arg.setGeneralParamCallbackArg.error,
		Arg.setGeneralParamCallbackArg.errorId);
}
void CDioLcAdapter::GetGeneralParamCB(LC_CALLBACK_ARG_T Arg)
{
	emit pThis->signalGetGeneralParamCB(Arg.getGeneralParamCallbackArg.paramId,
		Arg.getGeneralParamCallbackArg.paramValue,
		Arg.getGeneralParamCallbackArg.error,
		Arg.getGeneralParamCallbackArg.errorId);
}



