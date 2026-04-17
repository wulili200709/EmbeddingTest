#pragma once
#include <windows.h>
#include <QWidget>
#include <qserialport.h>
#include <qslider.h>
#include <qthread.h>
#include "ui_lightControlPage.h"
#include "dioLcAdapter.h"


class lightControlPage : public QWidget
{
	Q_OBJECT

public:

	lightControlPage(CDioLcAdapter * adapter, QWidget *parent = Q_NULLPTR);
	~lightControlPage();

signals:
	int signalBoardOpenPort(unsigned int devId, unsigned short port);
	int signalBoardClosePort(unsigned int devId, unsigned short port);
	int signalBoardIsDeviceOpened(unsigned char devId);

	int signalBoardGetVerInfo(unsigned int devId);

	int signalBoardSetPwmParams(unsigned int devId,
		unsigned char ucChIdx,
		unsigned char ucPwmMode,
		unsigned char ucPwmValue,
		unsigned char ucPwmHoldingTime,
		unsigned char ucPwmOnOff);

	int signalBoardGetPwmParams(unsigned int devId, unsigned char ucChIdx);

	int signalBoardGetDiStatus(unsigned int devId);
	int signalBoardSetDoStatus(unsigned int devId);

	int signalBoardPollingTrans(unsigned int devId,
		unsigned char ucDoStatus,
		unsigned char ucPwmOnOff,
		unsigned char ucReserve1,
		unsigned char ucReserve2);

	int signalBoardIAPDownload(unsigned int devId, QString imageFile);

	int signalSetGeneralParam(unsigned int devId, unsigned char paramId, unsigned char ucParamLen, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId);
	int signalGetGeneralParam(unsigned int devId, unsigned char paramId, unsigned char ucParmaLen, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId);


protected slots:

	void slotOnConnect();

	void slotOnReadParamCh0();
	void slotOnWriteParamCh0();
	void slotOnTurnOnCh0(bool onoff);
	void slotOnWriteParamOnlineCh0(int value);

	void slotOnReadParamCh1();
	void slotOnWriteParamCh1();
	void slotOnTurnOnCh1(bool onoff);
	void slotOnWriteParamOnlineCh1(int value);

	void slotOnReadParamCh2();
	void slotOnWriteParamCh2();
	void slotOnTurnOnCh2(bool onoff);
	void slotOnWriteParamOnlineCh2(int value);

	void slotOnReadParamCh3();
	void slotOnWriteParamCh3();
	void slotOnTurnOnCh3(bool onoff);
	void slotOnWriteParamOnlineCh3(int value);

	void slotDownloadDisplay();

	void slotOnGetCh0Advanced();
	void slotOnSetCh0Advanced();

	void slotOnGetCh1Advanced();
	void slotOnSetCh1Advanced();

	void slotOnGetCh2Advanced();
	void slotOnSetCh2Advanced();

	void slotOnGetCh3Advanced();
	void slotOnSetCh3Advanced();

	void slotComOpenCB(unsigned short portNum, 
		unsigned char hardwareMajorVer,
		unsigned char hardwareMinorVer,
		unsigned char hardwareRevVer,
		unsigned char firmwareMajorVer,
		unsigned char firmwareMinorVer,
		unsigned char firmwareRevVer,
		unsigned char ucError, unsigned int uiErrorId);
	void slotComCloseCB(unsigned char ucError, unsigned int uiErrorId);
	void slotGetDeviceVerCB(
		unsigned char  ucHardwareMajorVer,
		unsigned char  ucHardwareMinorVer,
		unsigned char  ucHardwareRevVer,
		unsigned char  ucFirmwareMajorVer,
		unsigned char  ucFirmwareMinorVer,
		unsigned char  ucFirmwareRevVer,
		unsigned char  ucError,
		unsigned int  uiErrorId);
	void slotSetPwmParamsCB(unsigned char  ucChIdx, unsigned char ucError, unsigned int uiErrorId);
	void slotGetPwmParamsCB(unsigned char  ucChIdx,
		unsigned char  ucPwmMode,
		unsigned char  ucPwmValue,
		unsigned char  ucPwmHoldingTime,
		unsigned char  ucPwmOnOff,
		unsigned char ucError,
		unsigned int uiErrorId);
	
	void slotTurnOnOffPwmSingleChannelCB(unsigned char  ucChIdx, unsigned char  ucStatus, unsigned char  ucError, unsigned int uiErrorId);
	void slotTurnOnOffPwmAllChannelCB(unsigned char  ucStatus, unsigned char ucError, unsigned int uiErrorId);
	void slotGetDiStatusCB(unsigned char  ucStatus, unsigned char ucError, unsigned int uiErrorId);
	void slotSetDoStatusCB(unsigned char  ucStatus, unsigned char ucError, unsigned int uiErrorId);

	void slotPollingTransCB(unsigned char ucDiStatus, unsigned char ucPwmStatus, unsigned char Reserve, unsigned char ucError, unsigned int uiErrorId);



	void slotSetGeneralParamCB(unsigned char paramId, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId);
	void slotGetGeneralParamCB(unsigned char paramId, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId);

	

	

private:
	void initSettings();

	int getComPort(QString comName);

	
private:
	Ui::lightControlPage ui;
	static lightControlPage *pThis;
	unsigned int m_downloadPercent;

	QSerialPort *m_currentPort;
	int m_port;
	uint m_baudrate;
	int m_databits;


	QTimer *m_timerDownloadFirmware;

	QString m_imageFilePath;
	int connect_flag;

	//lightControlServer *m_pServerThreadHdl;
	CDioLcAdapter *m_adapter;



};
