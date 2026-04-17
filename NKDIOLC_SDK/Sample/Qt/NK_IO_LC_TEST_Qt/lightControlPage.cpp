
#include <QSerialPortInfo>
#include <QSerialPort>
#include <qthread.h>
#include <qfiledialog.h>
#include <qtimer.h>
#include "lightControlPage.h"
//#include "PortSerial.h"
#include "qdebug.h"
#include "qmessagebox.h"

#define DEV_ID 0x01





void setTabWidgetStyleSheet(QTabWidget* obj) {
#if 1
	obj->setStyleSheet(
		"QTabWidget{"
		"background-color:transparent;" //transparent
		"}"
		"QTabWidget::pane{"
		"    border:2px;"
		//"    border-color: black;"
		//"border-style: outset;"
		"}"
		"QTabWidget::tab-bar{"
		"    alignment:left;"
		"    border:1px;"
		"    border-color: black;"
		"}"
		"QTabBar::tab{"
		"    background:#E4E4E4;"
		"    border:0px;"
		"    border-color: black;"
		"    color:black;"
		"    min-width:35ex;"
		"    min-height:8ex;"
		"}"
		"QTabBar::tab:hover{"
		"    background:#87CEFA;"
		"color:black;"
		"}"
		"QTabBar::tab:selected{"
		"    border-color: black;"
		"    background:#27A7F8;"
		"    color:black;"
		"}"
	);
#else
	obj->setStyleSheet(
		"QTabWidget::pane{ \
		border:none; \
} \
QTabWidget::tab - bar{ \
		alignment:left; \
} \
QTabBar::tab{ \
	background:transparent; \
	color:white; \
	min - width:30ex; \
	min - height:10ex; \
} \
QTabBar::tab:hover{ \
	background:rgb(255, 255, 255, 100); \
} \
QTabBar::tab : selected{ \
	border - color: white; \
	background:white; \
	color:green; \
}" );
#endif 
}


lightControlPage* lightControlPage::pThis = NULL;

lightControlPage::lightControlPage(CDioLcAdapter * adapter, QWidget *parent)
	: QWidget(parent)
	, m_adapter(adapter)
{
	ui.setupUi(this);
	setTabWidgetStyleSheet(ui.tabWidget);
	pThis = this;
	//s_downloadPercent = 0;
	
	//ui.m_comboBoxBaudRate->hide();
	//ui.m_lblBaudRate->hide();
	//ui.m_lblPort->hide();
	//ui.m_comboBoxPort->hide();
	

	//m_pServerThreadHdl = new lightControlServer();
	connect(this, SIGNAL(signalBoardOpenPort(unsigned int , unsigned short )), CDioLcAdapter::pThis,SLOT(slotBoardOpenPort(unsigned int, unsigned short)));
	connect(this, SIGNAL(signalBoardClosePort(unsigned int, unsigned short)), CDioLcAdapter::pThis, SLOT(slotBoardClosePort(unsigned int, unsigned short)));
	connect(this, SIGNAL(signalBoardIsDeviceOpened_t(unsigned char )), CDioLcAdapter::pThis, SLOT(slotBoardIsDeviceOpened(unsigned char)));
	connect(this, SIGNAL(signalBoardGetVerInfo(unsigned int )), CDioLcAdapter::pThis, SLOT(slotBoardGetVerInfo(unsigned int)));
	connect(this, SIGNAL(signalBoardSetPwmParams(unsigned int ,	unsigned char ,unsigned char ,unsigned char ,unsigned char ,unsigned char )), CDioLcAdapter::pThis, SLOT(slotBoardSetPwmParams(unsigned int, unsigned char, unsigned char, unsigned char, unsigned char, unsigned char)));
	connect(this, SIGNAL(signalBoardGetPwmParams(unsigned int , unsigned char )), CDioLcAdapter::pThis, SLOT(slotBoardGetPwmParams(unsigned int, unsigned char)));
	
	connect(this, SIGNAL(signalBoardIAPDownload(unsigned int , QString)), CDioLcAdapter::pThis, SLOT(slotBoardIAPDownload(unsigned int, QString)));


	connect(CDioLcAdapter::pThis, SIGNAL(signalComOpenCB(unsigned short, 
		unsigned char, 
		unsigned char , 
		unsigned char,
		unsigned char, 
		unsigned char, 
		unsigned char,
		unsigned char, 
		unsigned int)), this, SLOT(slotComOpenCB(unsigned short, 
			unsigned char, 
			unsigned char,
			unsigned char,
			unsigned char,
			unsigned char,
			unsigned char,
			unsigned char,
			unsigned int)));
	connect(CDioLcAdapter::pThis, SIGNAL(signalComCloseCB(unsigned char, unsigned int)), this, SLOT(slotComCloseCB(unsigned char , unsigned int )));
	connect(CDioLcAdapter::pThis, SIGNAL(signalGetDeviceVerCB(unsigned char,unsigned char,unsigned char,unsigned char,unsigned char,unsigned char,unsigned char,unsigned int)),
		this, SLOT(slotGetDeviceVerCB(unsigned char  ,unsigned char  ,unsigned char  ,unsigned char  ,unsigned char  ,unsigned char  ,unsigned char  ,unsigned int  )));
	connect(CDioLcAdapter::pThis, SIGNAL(signalSetPwmParamsCB(unsigned char, unsigned char, unsigned int)), this, SLOT(slotSetPwmParamsCB(unsigned char , unsigned char , unsigned int )));
	
	connect(CDioLcAdapter::pThis, SIGNAL(signalGetPwmParamsCB(unsigned char  ,
		unsigned char,
		unsigned char,
		unsigned char,
		unsigned char,
		unsigned char,
		unsigned int)), 
		this, SLOT(slotGetPwmParamsCB(unsigned char  ,
		unsigned char  ,
		unsigned char  ,
		unsigned char  ,
		unsigned char  ,
		unsigned char ,
		unsigned int )));

	connect(CDioLcAdapter::pThis, SIGNAL(signalIAPDownloadCB(unsigned char, unsigned int, unsigned int, unsigned char, unsigned int)),
		this, SLOT(slotIAPDownloadCB(unsigned char , unsigned int , unsigned int , unsigned char , unsigned int )));
	
	connect(CDioLcAdapter::pThis, SIGNAL(signalSetGeneralParamCB(unsigned char, unsigned int, unsigned char, unsigned int)),
		this, SLOT(slotSetGeneralParamCB(unsigned char, unsigned int, unsigned char, unsigned int)));
	connect(this, SIGNAL(signalSetGeneralParam(unsigned int, unsigned char, unsigned char, unsigned int, unsigned char, unsigned int)),
		CDioLcAdapter::pThis, SLOT(slotSetGeneralParam(unsigned int, unsigned char, unsigned char, unsigned int, unsigned char, unsigned int)));

	connect(CDioLcAdapter::pThis, SIGNAL(signalGetGeneralParamCB(unsigned char, unsigned int, unsigned char, unsigned int)),
		this, SLOT(slotGetGeneralParamCB(unsigned char, unsigned int, unsigned char, unsigned int)));
	connect(this, SIGNAL(signalGetGeneralParam(unsigned int, unsigned char, unsigned char, unsigned int, unsigned char, unsigned int)),
		CDioLcAdapter::pThis, SLOT(slotGetGeneralParam(unsigned int, unsigned char, unsigned char, unsigned int, unsigned char, unsigned int)));


	m_adapter->setPriority(QThread::TimeCriticalPriority);
	m_adapter->start();



	initSettings();
	
	
	

}

lightControlPage::~lightControlPage()
{
	
}



void lightControlPage::initSettings()
{

	ui.m_hardwareVer->setText("");
	ui.m_firmwareVer->setText("");
	ui.tabAdvanced->hide(); // for the advanced settings


	connect_flag = 0;
	ui.tabWidget->setEnabled(false);
	ui.m_connectBtn->setText(tr("Connect"));
	ui.m_connectBtn->setStyleSheet("background-color:red;");



	ui.m_comboBoxPort->clear();
	m_currentPort = new QSerialPort();
	QSerialPortInfo comPortInfo;
	foreach(const QSerialPortInfo &info, QSerialPortInfo::availablePorts())
	{
		comPortInfo = info;
		//ui.m_comboBoxPort->addItem(comPortInfo.portName());
	}
	//ui.m_comboBoxPort->addItem("COM6");
	QString comPort = "COM" + QString("%1").arg(m_adapter->m_port);
	ui.m_comboBoxPort->addItem(comPort);

	//QStringList ComList;
	//ComList << "COM1" << "COM2" << "COM3" << "COM4" << "COM5" << "COM6";
	//ui.m_comboBoxPort->addItems(ComList);
	//ui.m_comboBoxPort->setCurrentIndex(5);


	ui.m_comboBoxBaudRate->clear();
	QStringList baudRateList;
	baudRateList << "115200" << "57600" << "38400" << "19200" << "9600";
	ui.m_comboBoxBaudRate->addItems(baudRateList);
	ui.m_comboBoxBaudRate->setCurrentIndex(0);
	QStringList pwmModeList;
	pwmModeList << "Soft Switch" << "Hard Switch" << "Hard Trigger" << "Soft Trigger";


	ui.m_ch0Mode->clear();
	ui.m_ch0Mode->addItems(pwmModeList);

	ui.m_ch1Mode->clear();
	ui.m_ch1Mode->addItems(pwmModeList);

	ui.m_ch2Mode->clear();
	ui.m_ch2Mode ->addItems(pwmModeList);

	ui.m_ch3Mode->clear();
	ui.m_ch3Mode ->addItems(pwmModeList);

	ui.m_ch0Mode->setCurrentIndex(0);
	ui.m_ch1Mode->setCurrentIndex(0);
	ui.m_ch2Mode->setCurrentIndex(0);
	ui.m_ch3Mode->setCurrentIndex(0);

	QStringList holdingTimeUnitList;
	holdingTimeUnitList << "1000ms" << "100ms" << "10ms" << "1ms";

	ui.m_ch0HoldingTimeUnit->clear();
	ui.m_ch0HoldingTimeUnit->addItems(holdingTimeUnitList);

	ui.m_ch1HoldingTimeUnit->clear();
	ui.m_ch1HoldingTimeUnit->addItems(holdingTimeUnitList);

	ui.m_ch2HoldingTimeUnit->clear();
	ui.m_ch2HoldingTimeUnit->addItems(holdingTimeUnitList);

	ui.m_ch3HoldingTimeUnit->clear();
	ui.m_ch3HoldingTimeUnit->addItems(holdingTimeUnitList);

	connect(ui.m_connectBtn, SIGNAL(clicked()), this, SLOT(slotOnConnect()));

	connect(ui.m_ch0ReadBtn, SIGNAL(clicked()), this, SLOT(slotOnReadParamCh0()));
	connect(ui.m_ch0WriteBtn, SIGNAL(clicked()), this, SLOT(slotOnWriteParamCh0()));
	connect(ui.m_ch0Switch, SIGNAL(checkedChanged(bool)), this, SLOT(slotOnTurnOnCh0(bool)));
	connect(ui.m_ch0Slider, SIGNAL(valueChanged(int)), this, SLOT(slotOnWriteParamOnlineCh0(int)));

	connect(ui.m_ch1ReadBtn, SIGNAL(clicked()), this, SLOT(slotOnReadParamCh1()));
	connect(ui.m_ch1WriteBtn, SIGNAL(clicked()), this, SLOT(slotOnWriteParamCh1()));
	connect(ui.m_ch1Switch, SIGNAL(checkedChanged(bool)), this, SLOT(slotOnTurnOnCh1(bool)));
	connect(ui.m_ch1Slider, SIGNAL(valueChanged(int)), this, SLOT(slotOnWriteParamOnlineCh1(int)));

	connect(ui.m_ch2ReadBtn, SIGNAL(clicked()), this, SLOT(slotOnReadParamCh2()));
	connect(ui.m_ch2WriteBtn, SIGNAL(clicked()), this, SLOT(slotOnWriteParamCh2()));
	connect(ui.m_ch2Switch, SIGNAL(checkedChanged(bool)), this, SLOT(slotOnTurnOnCh2(bool)));
	connect(ui.m_ch2Slider, SIGNAL(valueChanged(int)), this, SLOT(slotOnWriteParamOnlineCh2(int)));

	connect(ui.m_ch3ReadBtn, SIGNAL(clicked()), this, SLOT(slotOnReadParamCh3()));
	connect(ui.m_ch3WriteBtn, SIGNAL(clicked()), this, SLOT(slotOnWriteParamCh3()));
	connect(ui.m_ch3Switch, SIGNAL(checkedChanged(bool)), this, SLOT(slotOnTurnOnCh3(bool)));
	connect(ui.m_ch3Slider, SIGNAL(valueChanged(int)), this, SLOT(slotOnWriteParamOnlineCh3(int)));

	connect(ui.m_ch0AdvReadBtn, SIGNAL(clicked()), this, SLOT(slotOnGetCh0Advanced()));
	connect(ui.m_ch0AdvWriteBtn, SIGNAL(clicked()), this, SLOT(slotOnSetCh0Advanced()));

	connect(ui.m_ch1AdvReadBtn, SIGNAL(clicked()), this, SLOT(slotOnGetCh1Advanced()));
	connect(ui.m_ch1AdvWriteBtn, SIGNAL(clicked()), this, SLOT(slotOnSetCh1Advanced()));

	connect(ui.m_ch2AdvReadBtn, SIGNAL(clicked()), this, SLOT(slotOnGetCh2Advanced()));
	connect(ui.m_ch2AdvWriteBtn, SIGNAL(clicked()), this, SLOT(slotOnSetCh2Advanced()));

	connect(ui.m_ch3AdvReadBtn, SIGNAL(clicked()), this, SLOT(slotOnGetCh3Advanced()));
	connect(ui.m_ch3AdvWriteBtn, SIGNAL(clicked()), this, SLOT(slotOnSetCh3Advanced()));

	// BaudRate
	if (ui.m_comboBoxBaudRate->currentText() == "115200")
	{
		m_currentPort->setBaudRate(QSerialPort::Baud115200, QSerialPort::AllDirections);
		m_baudrate = 115200;
	}
	else if (ui.m_comboBoxBaudRate->currentText() == "9600")
	{
		m_currentPort->setBaudRate(QSerialPort::Baud9600, QSerialPort::AllDirections);
		m_baudrate = 9600;
	}
	else if (ui.m_comboBoxBaudRate->currentText() == "1200")
	{
		m_currentPort->setBaudRate(QSerialPort::Baud1200, QSerialPort::AllDirections);
		m_baudrate = 1200;
	}
	else if (ui.m_comboBoxBaudRate->currentText() == "2400")
	{
		m_currentPort->setBaudRate(QSerialPort::Baud2400, QSerialPort::AllDirections);
		m_baudrate = 2400;
	}
	else if (ui.m_comboBoxBaudRate->currentText() == "4800")
	{
		m_currentPort->setBaudRate(QSerialPort::Baud4800, QSerialPort::AllDirections);
		m_baudrate = 4800;
	}
	else if (ui.m_comboBoxBaudRate->currentText() == "19200")
	{
		m_currentPort->setBaudRate(QSerialPort::Baud19200, QSerialPort::AllDirections);
		m_baudrate = 19200;
	}
	else if (ui.m_comboBoxBaudRate->currentText() == "38400")
	{
		m_currentPort->setBaudRate(QSerialPort::Baud38400, QSerialPort::AllDirections);
		m_baudrate = 38400;
	}
	else if (ui.m_comboBoxBaudRate->currentText() == "57600")
	{
		m_currentPort->setBaudRate(QSerialPort::Baud57600, QSerialPort::AllDirections);
		m_baudrate = 57600;
	}

	m_currentPort->setDataBits(QSerialPort::Data8);
	m_currentPort->setStopBits(QSerialPort::OneStop);
	m_currentPort->setParity(QSerialPort::NoParity);

	
	

}

int lightControlPage::getComPort(QString comName)
{
	int ret = -1;
	QString port = comName;
	//QString pattern("^[+-] ? \\d*[.] ? \\d*$");
	QString pattern("^[1-9]\\d*|0$");
	QRegExp rx(pattern);
	if (port != NULL)
	{
		//正则表达式剔除非数字字符（不包含小数点.）
		port = port.replace(QRegExp("[^\\d.\\d]"), "");
		if (rx.exactMatch(port))
		{
			ret = QString(port).toInt();
		}
	}
	return ret;
}

void lightControlPage::slotOnConnect()
{
	
	int retry = 0;
	int writeBytes = 0;
	int readBytes = 0;
	m_port = getComPort(ui.m_comboBoxPort->currentText());
	//m_port = 3;

	if (connect_flag == 0)
	{
		emit signalBoardOpenPort(DEV_ID, m_port);

		//emit signalBoardGetVerInfo(DEV_ID);

		

	}
	else if(connect_flag == 1)
	{
		//BoardClosePort(DEV_ID, m_port, this->ComCloseCB);
		emit signalBoardClosePort(DEV_ID, m_port);
	}


}

void lightControlPage::slotOnReadParamCh0()
{
	emit signalBoardGetPwmParams(DEV_ID, 0x01);
}
void lightControlPage::slotOnWriteParamCh0()
{

	emit signalBoardSetPwmParams(DEV_ID, 
		0x01, 
		ui.m_ch0Mode->currentIndex(), 
		ui.m_ch0Slider->value(), 
		ui.m_ch0PwmHoldingTime->value(), 
		(ui.m_ch0Switch->getChecked() == true) ? 1 : 0);

}
void lightControlPage::slotOnTurnOnCh0(bool onoff)
{
	emit signalBoardSetPwmParams(DEV_ID,
		0x01,
		ui.m_ch0Mode->currentIndex(),
		ui.m_ch0Slider->value(),
		ui.m_ch0PwmHoldingTime->value(),
		(onoff == true) ? 1 : 0);
}
void lightControlPage::slotOnWriteParamOnlineCh0(int value)
{
	if (ui.m_ch0Switch->getChecked() == true)
	{
		emit signalBoardSetPwmParams(DEV_ID,
			0x01,
			ui.m_ch0Mode->currentIndex(),
			value,
			ui.m_ch0PwmHoldingTime->value(),
			 1 );
	}
}

void lightControlPage::slotOnGetCh0Advanced()
{
	// get the holding time unit
	emit signalGetGeneralParam(DEV_ID, 10, 1, 0, 0, 0);
}
void lightControlPage::slotOnSetCh0Advanced()
{
	// set the holding time unit
	unsigned char ParamId = 10;
	unsigned char ParamLen = 1;
	unsigned int ParamVal = ui.m_ch0HoldingTimeUnit->currentIndex();

	emit signalSetGeneralParam(DEV_ID, ParamId, ParamLen, ParamVal, 0, 0);
}


void lightControlPage::slotOnReadParamCh1()
{
	emit signalBoardGetPwmParams(DEV_ID, 0x02);
}
void lightControlPage::slotOnWriteParamCh1()
{

	emit signalBoardSetPwmParams(DEV_ID,
		0x02,
		ui.m_ch1Mode->currentIndex(),
		ui.m_ch1Slider->value(),
		ui.m_ch1PwmHoldingTime->value(),
		(ui.m_ch1Switch->getChecked() == true) ? 1 : 0);
 
	
}
void lightControlPage::slotOnTurnOnCh1(bool onoff)
{

	emit signalBoardSetPwmParams(DEV_ID,
		0x02,
		ui.m_ch1Mode->currentIndex(),
		ui.m_ch1Slider->value(),
		ui.m_ch1PwmHoldingTime->value(),
		(onoff == true) ? 1 : 0);

}

void lightControlPage::slotOnWriteParamOnlineCh1(int value)
{
	if (ui.m_ch1Switch->getChecked() == true)
	{
		emit signalBoardSetPwmParams(DEV_ID,
			0x02,
			ui.m_ch1Mode->currentIndex(),
			value,
			ui.m_ch1PwmHoldingTime->value(),
			1);
	}
}

void lightControlPage::slotOnGetCh1Advanced()
{
	emit signalGetGeneralParam(DEV_ID, 11, 1, 0, 0, 0);
}
void lightControlPage::slotOnSetCh1Advanced()
{
	unsigned char ParamId = 11;
	unsigned char ParamLen = 1;
	unsigned int ParamVal = ui.m_ch1HoldingTimeUnit->currentIndex();
	emit signalSetGeneralParam(DEV_ID, ParamId, ParamLen, ParamVal, 0, 0);
}

void lightControlPage::slotOnReadParamCh2()
{
	emit signalBoardGetPwmParams(DEV_ID, 0x04);
}
void lightControlPage::slotOnWriteParamCh2()
{

	emit signalBoardSetPwmParams(DEV_ID,
		0x04,
		ui.m_ch2Mode->currentIndex(),
		ui.m_ch2Slider->value(),
		ui.m_ch2PwmHoldingTime->value(),
		(ui.m_ch2Switch->getChecked() == true) ? 1 : 0);

}
void lightControlPage::slotOnTurnOnCh2(bool onoff)
{

	emit signalBoardSetPwmParams(DEV_ID,
		0x04,
		ui.m_ch2Mode->currentIndex(),
		ui.m_ch2Slider->value(),
		ui.m_ch2PwmHoldingTime->value(),
		(onoff == true) ? 1 : 0);

}

void lightControlPage::slotOnWriteParamOnlineCh2(int value)
{
	if (ui.m_ch2Switch->getChecked() == true)
	{
		emit signalBoardSetPwmParams(DEV_ID,
			0x04,
			ui.m_ch2Mode->currentIndex(),
			value,
			ui.m_ch2PwmHoldingTime->value(),
			1);
	}
}

void lightControlPage::slotOnGetCh2Advanced()
{
	emit signalGetGeneralParam(DEV_ID, 12, 1, 0, 0, 0);
}
void lightControlPage::slotOnSetCh2Advanced()
{
	unsigned char ParamId = 12;
	unsigned char ParamLen = 1;
	unsigned int ParamVal = ui.m_ch2HoldingTimeUnit->currentIndex();

	emit signalSetGeneralParam(DEV_ID, ParamId, ParamLen, ParamVal, 0, 0);
}

void lightControlPage::slotOnReadParamCh3()
{
	emit signalBoardGetPwmParams(DEV_ID, 0x08);
}
void lightControlPage::slotOnWriteParamCh3()
{

	emit signalBoardSetPwmParams(DEV_ID,
		0x08,
		ui.m_ch3Mode->currentIndex(),
		ui.m_ch3Slider->value(),
		ui.m_ch3PwmHoldingTime->value(),
		(ui.m_ch3Switch->getChecked() == true) ? 1 : 0);

}
void lightControlPage::slotOnTurnOnCh3(bool onoff)
{

	emit signalBoardSetPwmParams(DEV_ID,
		0x08,
		ui.m_ch3Mode->currentIndex(),
		ui.m_ch3Slider->value(),
		ui.m_ch3PwmHoldingTime->value(),
		(onoff == true) ? 1 : 0);

}

void lightControlPage::slotOnWriteParamOnlineCh3(int value)
{
	if (ui.m_ch3Switch->getChecked() == true)
	{
		emit signalBoardSetPwmParams(DEV_ID,
			0x08,
			ui.m_ch3Mode->currentIndex(),
			value,
			ui.m_ch3PwmHoldingTime->value(),
			1);
	}
}

void lightControlPage::slotOnGetCh3Advanced()
{
	emit signalGetGeneralParam(DEV_ID, 13, 1, 0, 0, 0);
}
void lightControlPage::slotOnSetCh3Advanced()
{
	unsigned char ParamId = 13;
	unsigned char ParamLen = 1;
	unsigned int ParamVal = ui.m_ch3HoldingTimeUnit->currentIndex();

	emit signalSetGeneralParam(DEV_ID, ParamId, ParamLen, ParamVal, 0, 0);
}

void lightControlPage::slotDownloadDisplay()
{

}

void lightControlPage::slotComOpenCB(unsigned short portNum, 
	unsigned char hardwareMajorVer,
	unsigned char hardwareMinorVer,
	unsigned char hardwareRevVer,
	unsigned char firmwareMajorVer,
	unsigned char firmwareMinorVer,
	unsigned char firmwareRevVer, 
	unsigned char ucError, 
	unsigned int uiErrorId)
{
	//QMessageBox::information(this, tr("Open Port"), tr("Port Opened"));
	if (ucError)
	{
	}
	else
	{
		connect_flag = 1;
		ui.tabWidget->setEnabled(true);
		ui.m_connectBtn->setText(tr("Disconnect"));
		ui.m_connectBtn->setStyleSheet("background-color:green;");

		ui.m_hardwareVer->setText(QString("%1.%2.%3").arg(hardwareMajorVer).arg(hardwareMinorVer).arg(hardwareRevVer));
		ui.m_firmwareVer->setText(QString("%1.%2.%3").arg(firmwareMajorVer).arg(firmwareMinorVer).arg(firmwareRevVer));

		emit signalBoardGetPwmParams(DEV_ID, 0x01);
		emit signalBoardGetPwmParams(DEV_ID, 0x02);
		emit signalBoardGetPwmParams(DEV_ID, 0x04);
		emit signalBoardGetPwmParams(DEV_ID, 0x08);
	}
	
}
void lightControlPage::slotComCloseCB(unsigned char ucError, unsigned int uiErrorId)
{
	if (ucError)
	{
		ui.m_hardwareVer->setText("x.x.x");
		ui.m_firmwareVer->setText("x.x.x");
	}
	else
	{
		connect_flag = 0;
		ui.tabWidget->setEnabled(false);
		ui.m_connectBtn->setText(tr("Connect"));
		ui.m_connectBtn->setStyleSheet("background-color:red;");

		ui.m_hardwareVer->setText("");
		ui.m_firmwareVer->setText("");

	}
}
void lightControlPage::slotGetDeviceVerCB(
	unsigned char  ucHardwareMajorVer,
	unsigned char  ucHardwareMinorVer,
	unsigned char  ucHardwareRevVer,
	unsigned char  ucFirmwareMajorVer,
	unsigned char  ucFirmwareMinorVer,
	unsigned char  ucFirmwareRevVer,
	unsigned char  ucError,
	unsigned int  uiErrorId)
{
	if (ucError)
	{
		QMessageBox::warning(this, tr("Warning"),
			tr("Get device version Failed!"),
			QMessageBox::Ok);
		ui.m_hardwareVer->setText("x.x.x");
		ui.m_firmwareVer->setText("x.x.x");
	}
	else
	{
		ui.m_hardwareVer->setText(QString("%1.%2.%3").arg(ucHardwareMajorVer).arg(ucHardwareMinorVer).arg(ucHardwareRevVer));
		ui.m_firmwareVer->setText(QString("%1.%2.%3").arg(ucFirmwareMajorVer).arg(ucFirmwareMinorVer).arg(ucFirmwareRevVer));
	}
	
}
void lightControlPage::slotSetPwmParamsCB(unsigned char  ucChIdx, unsigned char ucError, unsigned int uiErrorId)
{

}
void lightControlPage::slotGetPwmParamsCB(unsigned char  ucChIdx,
	unsigned char  ucPwmMode,
	unsigned char  ucPwmValue,
	unsigned char  ucPwmHoldingTime,
	unsigned char  ucPwmOnOff,
	unsigned char ucError,
	unsigned int uiErrorId)
{
	if (ucError)
	{
		QMessageBox::warning(this, tr("Warning"),
			tr("Get Pwm params error!"),
			QMessageBox::Ok);
	}
	else
	{
		switch (ucChIdx)
		{
		case 0x01:

			ui.m_ch0Mode->setCurrentIndex(ucPwmMode);
			ui.m_ch0Slider->setValue(ucPwmValue);
			ui.m_ch0PwmHoldingTime->setValue(ucPwmHoldingTime);
			ui.m_ch0Switch->setChecked((ucPwmOnOff & 0x01) == 0x01 ? true : false);

			break;
		case 0x02:

			ui.m_ch1Mode->setCurrentIndex(ucPwmMode);
			ui.m_ch1Slider->setValue(ucPwmValue);
			ui.m_ch1PwmHoldingTime->setValue(ucPwmHoldingTime);
			ui.m_ch1Switch->setChecked((ucPwmOnOff & 0x01) == 0x01 ? true : false);

			break;
		case 0x04:

			ui.m_ch2Mode->setCurrentIndex(ucPwmMode);
			ui.m_ch2Slider->setValue(ucPwmValue);
			ui.m_ch2PwmHoldingTime->setValue(ucPwmHoldingTime);
			ui.m_ch2Switch->setChecked((ucPwmOnOff & 0x01) == 0x01 ? true : false);

			break;
		case 0x08:

			ui.m_ch3Mode->setCurrentIndex(ucPwmMode);
			ui.m_ch3Slider->setValue(ucPwmValue);
			ui.m_ch3PwmHoldingTime->setValue(ucPwmHoldingTime);
			ui.m_ch3Switch->setChecked((ucPwmOnOff & 0x01) == 0x01 ? true : false);

			break;
		default:
			break;
		}
	}
}

void lightControlPage::slotTurnOnOffPwmSingleChannelCB(unsigned char  ucChIdx, unsigned char  ucStatus, unsigned char  ucError, unsigned int uiErrorId)
{

}
void lightControlPage::slotTurnOnOffPwmAllChannelCB(unsigned char  ucStatus, unsigned char ucError, unsigned int uiErrorId)
{

}
void lightControlPage::slotGetDiStatusCB(unsigned char  ucStatus, unsigned char ucError, unsigned int uiErrorId)
{

}
void lightControlPage::slotSetDoStatusCB(unsigned char  ucStatus, unsigned char ucError, unsigned int uiErrorId)
{

}

void lightControlPage::slotPollingTransCB(unsigned char ucDiStatus, unsigned char ucPwmStatus, unsigned char Reserve, unsigned char ucError, unsigned int uiErrorId)
{

}

void lightControlPage::slotSetGeneralParamCB(unsigned char paramId, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId)
{
	if (ucError > 0)
	{
		QMessageBox::warning(this, tr("Warning"),
			tr("Set Advanced parameter error!"),
			QMessageBox::Ok);
	}
	else
	{
		unsigned int timeUnit = 1000;
		if (paramId >= 10 && paramId <= 13)
		{
			switch (paramValue)
			{
			case 0:
				timeUnit = 1000;
				break;
			case 1:
				timeUnit = 100;
				break;
			case 2:
				timeUnit = 10;
				break;
			case 3:
				timeUnit = 1;
				break;
			default:
				timeUnit = 1000;
				break;
			}
		}

		switch (paramId)
		{
		case 10:
			ui.m_lblCh0HoldingTimeUnitCurrentValue->setText(QString::number(timeUnit) + "ms");
			break;
		case 11:
			ui.m_lblCh1HoldingTimeUnitCurrentValue->setText(QString::number(timeUnit) + "ms");
			break;
		case 12:
			ui.m_lblCh2HoldingTimeUnitCurrentValue->setText(QString::number(timeUnit) + "ms");
			break;
		case 13:
			ui.m_lblCh3HoldingTimeUnitCurrentValue->setText(QString::number(timeUnit) + "ms");
			break;
		default:
			break;
		}
	}

}
void lightControlPage::slotGetGeneralParamCB(unsigned char paramId, unsigned int paramValue, unsigned char ucError, unsigned int uiErrorId)
{
	if (ucError > 0)
	{
		QMessageBox::warning(this, tr("Warning"),
			tr("Get Advanced parameter error!"),
			QMessageBox::Ok);
	}
	else
	{
		unsigned int timeUnit = 1000;
		if (paramId >= 10 && paramId <= 13)
		{
			switch (paramValue)
			{
			case 0:
				timeUnit = 1000;
				break;
			case 1:
				timeUnit = 100;
				break;
			case 2:
				timeUnit = 10;
				break;
			case 3:
				timeUnit = 1;
				break;
			default:
				timeUnit = 1000;
				break;
			}
		}

		switch (paramId)
		{
		case 10:
			ui.m_lblCh0HoldingTimeUnitCurrentValue->setText(QString::number(timeUnit) + "ms");
			ui.m_ch0HoldingTimeUnit->setCurrentIndex(paramValue);
			break;
		case 11:
			ui.m_lblCh1HoldingTimeUnitCurrentValue->setText(QString::number(timeUnit) + "ms");
			ui.m_ch1HoldingTimeUnit->setCurrentIndex(paramValue);
			break;
		case 12:
			ui.m_lblCh2HoldingTimeUnitCurrentValue->setText(QString::number(timeUnit) + "ms");
			ui.m_ch2HoldingTimeUnit->setCurrentIndex(paramValue);
			break;
		case 13:
			ui.m_lblCh3HoldingTimeUnitCurrentValue->setText(QString::number(timeUnit) + "ms");
			ui.m_ch3HoldingTimeUnit->setCurrentIndex(paramValue);
			break;
		default:
			break;
		}
	}
}

