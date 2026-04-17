#pragma once
#include <Windows.h> 
#include "estl.h"
#include <QWidget>
#include "ui_aioTestPage.h"
#include "DioLcAdapter.h"
#include "qtimer.h"
#include "qsettings.h"
#include "qmodbusrtuserialmaster.h"


#define HOLDING_REG_BASE_OFF        0
#define DEVID_REG_OFF              ( HOLDING_REG_BASE_OFF + 0 )
#define HARDWAREVER_REG_OFF        ( HOLDING_REG_BASE_OFF + 1 )
#define FIRMWARE_REG_OFF           ( HOLDING_REG_BASE_OFF + 2 )
#define MODBUSMODE_REG_OFF         ( HOLDING_REG_BASE_OFF + 3 )
#define MODBUSADDR_REG_OFF         ( HOLDING_REG_BASE_OFF + 4 )
#define MODBUSBAUD_REG_OFF         ( HOLDING_REG_BASE_OFF + 5 )
#define MODBUSCRC_REG_OFF          ( HOLDING_REG_BASE_OFF + 6 )
#define FILTERTYPE_REG_OFF         ( HOLDING_REG_BASE_OFF + 7 )
#define FILTERDEPTH_REG_OFF        ( HOLDING_REG_BASE_OFF + 8 )
#define DATAOUTTYPE_REG_OFF        ( HOLDING_REG_BASE_OFF + 9 )
#define SAMPLETYPE_CH1_REG_OFF     ( HOLDING_REG_BASE_OFF + 10 )
#define SAMPLETYPE_CH2_REG_OFF     ( HOLDING_REG_BASE_OFF + 11 )

#define CURRENT_SAMPLE_COMP_CH1_OFF   ( HOLDING_REG_BASE_OFF + 18 )
#define CURRENT_SAMPLE_COMP_CH2_OFF   ( HOLDING_REG_BASE_OFF + 19 )

#define VALUE_CH1_REG_OFF           ( HOLDING_REG_BASE_OFF + 30 )
#define VALUE_CH2_REG_OFF           ( HOLDING_REG_BASE_OFF + 31 )


class aioTestPage : public QWidget
{
	Q_OBJECT

public:
	aioTestPage(CDioLcAdapter * adapter, QWidget *parent = Q_NULLPTR);
	~aioTestPage();

private slots:
	void slotOnConnectClick(void);

	void slotOnReadDeviceInfoClick(void);
	void slotOnReadDeviceHardwareVerReady(void);
	void slotOnReadDeviceFirmwareVerReady(void);


	void slotOnReadCommonClick(void);
	void slotOnReadCommonReady(void);
	void slotOnReadFilterTypeReady(void);
	void slotOnReadFilterDepthReady(void);
	void slotOnReadOutputDataTypeReady(void);

	void slotOnWriteCommonClick(void);
	void slotOnWriteCommonReady(void);
	void slotOnWriteFilterTypeReady(void);
	void slotOnWriteFilterDepthReady(void);
	void slotOnWriteOutputDataTypeReady(void);

	void slotOnReadCh1Click(void);
	void slotOnReadCh1Ready(void);
	void slotOnReadSampleTypeCh1Ready(void);
	void slotOnReadCurrentCompCh1Ready(void);
	void slotOnReadValueCh1Ready(void);

	void slotOnWriteCh1Click(void);
	void slotOnWriteCh1Ready(void);
	void slotOnWriteSampleTypeCh1Ready(void);
	void slotOnWriteCurrentCompCh1Ready(void);

	void slotOnReadCh2Click(void);
	void slotOnReadCh2Ready(void);
	void slotOnReadSampleTypeCh2Ready(void);
	void slotOnReadCurrentCompCh2Ready(void);
	void slotOnReadValueCh2Ready(void);

	void slotOnWriteCh2Click(void);
	void slotOnWriteCh2Ready(void);
	void slotOnWriteSampleTypeCh2Ready(void);
	void slotOnWriteCurrentCompCh2Ready(void);

	void slotOnMasterUpdate(void);
	void slotonMasterUpdateReady(void);

private:
	void findCom(void);

private:
	Ui::aioTestPage ui;

	QTimer *m_masterTimer;
	QModbusReply *m_lastRequest;
	QModbusClient *m_master;
	
	int m_connectState;

	int m_parity;
	int m_baud;
	int m_dataBits;
	int m_stopBits;

	int m_responseTime;
	int m_numberOfRetries;

};
