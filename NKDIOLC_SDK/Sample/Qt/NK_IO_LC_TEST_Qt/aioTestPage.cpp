#include<Windows.h>
#include <QtSerialPort/QSerialPort>
#include <QtSerialPort/QSerialPortInfo>
#include <qmessagebox.h>
#include "aioTestPage.h"

#define SLAVE_ID  0x01


// Aio test page
aioTestPage::aioTestPage(CDioLcAdapter * adapter, QWidget *parent)
	: QWidget(parent)
	, m_connectState(0)
{
	ui.setupUi(this);
	m_master = new QModbusRtuSerialMaster(this);
	ui.m_comboBoxComType->clear();
	ui.m_comboBoxComType->addItem(tr("Modbus RTU"));
	ui.m_comboBoxComType->setCurrentIndex(0);
	findCom();
	QStringList baudRateList;
	baudRateList << "115200" << "57600" << "38400" << "19200" << "9600";
	ui.m_comboBoxBaudRate->addItems(baudRateList);
	ui.m_comboBoxBaudRate->setCurrentIndex(0);

	m_parity = QSerialPort::NoParity;
	m_baud = QSerialPort::Baud115200;
	m_dataBits = QSerialPort::Data8;
	m_stopBits = QSerialPort::OneStop;

	m_responseTime = 1000;
	m_numberOfRetries = 3;


	ui.m_lblHardwareVerValue->setText("xxxx");
	ui.m_lblFirmwareVerValue->setText("xxxx");


	QStringList filterTypeList;
	filterTypeList << "0:None filter" << "1:Mean filter";
	ui.m_cmboxFilterType->clear();
	ui.m_cmboxFilterType->addItems(filterTypeList);
	ui.m_cmboxFilterType->setCurrentIndex(0);

	QStringList outputDataTypeList;
	outputDataTypeList << "0:Orignal" << "1:AD" << "2:Current";
	ui.m_cmboxOutputDataType->clear();
	ui.m_cmboxOutputDataType->addItems(outputDataTypeList);
	ui.m_cmboxOutputDataType->setCurrentIndex(0);

	QStringList sampleTypeList;
	sampleTypeList << "0x0101:-10V-+10V" << "0x0102:-5V-+5V" << "0x0201:0-20mA" << "0x0202:4-20mA" << "0x0203:-20mA-+20mA";
	ui.m_cmboxSampleTypeCH1->clear();
	ui.m_cmboxSampleTypeCH1->addItems(sampleTypeList);
	ui.m_cmboxSampleTypeCH2->clear();
	ui.m_cmboxSampleTypeCH2->addItems(sampleTypeList);

	connect(ui.m_btnReadCommon, SIGNAL(clicked()), this, SLOT(slotOnReadCommonClick()));
	connect(ui.m_btnWriteCommon, SIGNAL(clicked()), this, SLOT(slotOnWriteCommonClick()));
	connect(ui.m_btnReadCh1, SIGNAL(clicked()),this, SLOT(slotOnReadCh1Click()));
	connect(ui.m_btnWriteCh1, SIGNAL(clicked()), this, SLOT(slotOnWriteCh1Click()));
	connect(ui.m_btnReadCh2, SIGNAL(clicked()), this, SLOT(slotOnReadCh2Click()));
	connect(ui.m_btnWriteCh2, SIGNAL(clicked()), this, SLOT(slotOnWriteCh2Click()));

	ui.m_btnConnect->setStyleSheet(QString("background-color:") + "red");
	connect(ui.m_btnConnect, SIGNAL(clicked()), this, SLOT(slotOnConnectClick()));

	m_masterTimer = new QTimer;
	m_masterTimer->setInterval(50);
	connect(m_masterTimer, SIGNAL(timeout()), this, SLOT(slotOnMasterUpdate()));
	m_masterTimer->start();
}

aioTestPage::~aioTestPage()
{
}


void aioTestPage::findCom(void)
{
	this->ui.m_comboBoxPort->clear();
	QStringList CommPortList;
	foreach(const QSerialPortInfo &info, QSerialPortInfo::availablePorts())
	{
		QSerialPort serial;
		serial.setPort(info);
		if (serial.open(QIODevice::ReadWrite))
		{
			CommPortList.append(serial.portName());
			serial.close();
		}
	}
	this->ui.m_comboBoxPort->addItems(CommPortList);
	this->ui.m_comboBoxPort->setCurrentIndex(0);
}



// slots
void aioTestPage::slotOnConnectClick(void)
{
	if (!m_master)
	{
		return;
	}
	else if (m_master->state() == QModbusDevice::ConnectedState)
	{
		m_master->disconnectDevice();
		ui.m_btnConnect->setText(tr("Connect"));
		m_connectState = 0;
		ui.m_btnConnect->setStyleSheet(QString("background-color:") + "red");
	}
	else if (m_master->state() != QModbusDevice::ConnectedState)
	{
		QVariant port;
		QString szPort = ui.m_comboBoxPort->currentText();
		port.setValue(szPort);
		m_master->setConnectionParameter(QModbusDevice::SerialPortNameParameter, port);

		m_master->setConnectionParameter(QModbusDevice::SerialBaudRateParameter, m_baud);
		m_master->setConnectionParameter(QModbusDevice::SerialParityParameter, m_parity);
		m_master->setConnectionParameter(QModbusDevice::SerialDataBitsParameter, m_dataBits);
		m_master->setConnectionParameter(QModbusDevice::SerialStopBitsParameter, m_stopBits);

		m_master->setTimeout(m_responseTime);
		m_master->setNumberOfRetries(m_numberOfRetries);
		if (!m_master->connectDevice())
		{
			//emit sendErrorEvent(CommError::ConnectionError);
			//connectAction->setEnabled(true);
			ui.m_btnConnect->setText(tr("Connect"));
			m_connectState = 0;
			ui.m_btnConnect->setStyleSheet(QString("background-color:") + "red");
			///disconnectAction->setEnabled(false);
			ui.m_lblHardwareVerValue->setText("xxxx");
			ui.m_lblFirmwareVerValue->setText("xxxx");

		}
		else
		{
			ui.m_btnConnect->setText(tr("Disconnect"));
			ui.m_btnConnect->setStyleSheet(QString("background-color:") + "green");
			m_connectState = 1;
			slotOnReadDeviceInfoClick();
			slotOnReadCommonClick();
			//slotOnReadCh1Click();
			//slotOnReadCh2Click();
		}

	}
	return;

	
}


void aioTestPage::slotOnReadDeviceInfoClick(void)
{
	if (!m_master)
		return;
	if (m_master->state() == QModbusDevice::ConnectedState)
	{
		QModbusDataUnit readHardwareVerUnit = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, HARDWAREVER_REG_OFF, 1);

		if (auto *reply = m_master->sendReadRequest(readHardwareVerUnit, SLAVE_ID))
		{
			if (!reply->isFinished())
			{
				connect(reply, SIGNAL(finished()), this, SLOT(slotOnReadDeviceHardwareVerReady()));
			}
			else
			{
				delete reply;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}

		QModbusDataUnit readFirmwareVerUnit = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, FIRMWARE_REG_OFF, 1);
		if (auto *reply1 = m_master->sendReadRequest(readFirmwareVerUnit, SLAVE_ID))
		{
			if (!reply1->isFinished())
			{
				connect(reply1, SIGNAL(finished()), this, SLOT(slotOnReadDeviceFirmwareVerReady()));
			}
			else
			{
				delete reply1;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}
	}
}


void aioTestPage::slotOnReadDeviceHardwareVerReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

		ui.m_lblHardwareVerValue->setText(QString::number(unit.value(0),16));
	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}


void aioTestPage::slotOnReadDeviceFirmwareVerReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

		ui.m_lblFirmwareVerValue->setText(QString::number(unit.value(0), 16));
	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}




void aioTestPage::slotOnReadCommonClick(void)
{
	if (!m_master)
		return;
	if (m_master->state() == QModbusDevice::ConnectedState)
	{
		QModbusDataUnit readFilterTypeUnit = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, FILTERTYPE_REG_OFF, 1);
		
		if (auto *reply = m_master->sendReadRequest(readFilterTypeUnit, SLAVE_ID))
		{
			if (!reply->isFinished())
			{
				connect(reply, SIGNAL(finished()), this, SLOT(slotOnReadFilterTypeReady()));
			}
			else
			{
				delete reply;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}

		QModbusDataUnit readFilterDepthUnit = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, FILTERDEPTH_REG_OFF, 1);
		if (auto *reply1 = m_master->sendReadRequest(readFilterDepthUnit, SLAVE_ID))
		{
			if (!reply1->isFinished())
			{
				connect(reply1, SIGNAL(finished()), this, SLOT(slotOnReadFilterDepthReady()));
			}
			else
			{
				delete reply1;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}

		QModbusDataUnit readOutputDataTypeUnit = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, DATAOUTTYPE_REG_OFF, 1);
		if (auto *reply2 = m_master->sendReadRequest(readOutputDataTypeUnit, SLAVE_ID))
		{
			if (!reply2->isFinished())
			{
				connect(reply2, SIGNAL(finished()), this, SLOT(slotOnReadOutputDataTypeReady()));
			}
			else
			{
				delete reply2;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}

	}
}

void aioTestPage::slotOnReadCommonReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();
		
		if (unit.valueCount() == 3)
		{
			
			ui.m_cmboxFilterType->setCurrentIndex(unit.value(0));
			ui.m_spinBoxFilterDepth->setValue(unit.value(1));
			ui.m_cmboxOutputDataType->setCurrentIndex(unit.value(2));

		}

	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}

void aioTestPage::slotOnReadFilterTypeReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

		ui.m_cmboxFilterType->setCurrentIndex(unit.value(0));
	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}
void aioTestPage::slotOnReadFilterDepthReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();
		ui.m_spinBoxFilterDepth->setValue(unit.value(0));

	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}
void aioTestPage::slotOnReadOutputDataTypeReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();
		ui.m_cmboxOutputDataType->setCurrentIndex(unit.value(0));

	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}


void aioTestPage::slotOnWriteCommonClick(void)
{
	if (!m_master)
		return;
	if (m_master->state() == QModbusDevice::ConnectedState)
	{
		QModbusDataUnit writeFilterTypeUnit = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, FILTERTYPE_REG_OFF, 1);
		writeFilterTypeUnit.setValue(0, (quint16)ui.m_cmboxFilterType->currentIndex());
		
		if (auto *reply = m_master->sendWriteRequest(writeFilterTypeUnit, SLAVE_ID))
		{
			if (!reply->isFinished())
			{
				connect(reply, SIGNAL(finished()), this, SLOT(slotOnWriteFilterTypeReady()));
			}
			else
			{
				delete reply; // broadcast replies return immediately
			}
		}
		QModbusDataUnit writeFilterDepthUnit = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, FILTERDEPTH_REG_OFF, 1);
		writeFilterDepthUnit.setValue(0, (quint16)ui.m_spinBoxFilterDepth->value());
		if (auto *reply2 = m_master->sendWriteRequest(writeFilterDepthUnit, SLAVE_ID))
		{
			if (!reply2->isFinished())
			{
				connect(reply2, SIGNAL(finished()), this, SLOT(slotOnWriteFilterDepthReady()));
			}
			else
			{
				delete reply2; // broadcast replies return immediately
			}
		}

		QModbusDataUnit writeOutputDataTypeUnit = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, DATAOUTTYPE_REG_OFF, 1);
		writeOutputDataTypeUnit.setValue(0, (quint16)ui.m_cmboxOutputDataType->currentIndex());
		if (auto *reply3 = m_master->sendWriteRequest(writeOutputDataTypeUnit, SLAVE_ID))
		{
			if (!reply3->isFinished())
			{
				connect(reply3, SIGNAL(finished()), this, SLOT(slotOnWriteOutputDataTypeReady()));
			}
			else
			{
				delete reply3; // broadcast replies return immediately
			}
		}
	}

}
void aioTestPage::slotOnWriteCommonReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();
		
	}
	else if (reply->error() == QModbusDevice::ProtocolError)
	{
		
	}
	else 
	{
		
	}

	reply->deleteLater();
}

void aioTestPage::slotOnWriteFilterTypeReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

	}
	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}

	reply->deleteLater();
}
void aioTestPage::slotOnWriteFilterDepthReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

	}
	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}

	reply->deleteLater();
}
void aioTestPage::slotOnWriteOutputDataTypeReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

	}
	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}

	reply->deleteLater();
}


void aioTestPage::slotOnReadCh1Click(void)
{
	if (!m_master)
		return;
	if (m_master->state() == QModbusDevice::ConnectedState)
	{
		if (auto *reply = m_master->sendReadRequest(QModbusDataUnit(QModbusDataUnit::HoldingRegisters, SAMPLETYPE_CH1_REG_OFF, 1), SLAVE_ID))
		{
			if (!reply->isFinished())
			{
				connect(reply, SIGNAL(finished()), this, SLOT(slotOnReadSampleTypeCh1Ready()));
			}
			else
			{
				delete reply;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}

		if (auto *reply2 = m_master->sendReadRequest(QModbusDataUnit(QModbusDataUnit::HoldingRegisters, CURRENT_SAMPLE_COMP_CH1_OFF, 1), SLAVE_ID))
		{
			if (!reply2->isFinished())
			{
				connect(reply2, SIGNAL(finished()), this, SLOT(slotOnReadCurrentCompCh1Ready()));
			}
			else
			{
				delete reply2;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}

		if (auto *reply3 = m_master->sendReadRequest(QModbusDataUnit(QModbusDataUnit::HoldingRegisters, VALUE_CH1_REG_OFF, 1), SLAVE_ID))
		{
			if (!reply3->isFinished())
			{
				connect(reply3, SIGNAL(finished()), this, SLOT(slotOnReadValueCh1Ready()));
			}
			else
			{
				delete reply3;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}
	}
}
void aioTestPage::slotOnReadCh1Ready(void)
{

}

void aioTestPage::slotOnReadSampleTypeCh1Ready(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

		if (unit.valueCount() == 1)
		{

			switch (unit.value(0))
			{
			case 0x0101:
				ui.m_cmboxSampleTypeCH1->setCurrentIndex(0);
				break;
			case 0x0102:
				ui.m_cmboxSampleTypeCH1->setCurrentIndex(1);
				break;
			case 0x0201:
				ui.m_cmboxSampleTypeCH1->setCurrentIndex(2);
				break;
			case 0x0202:
				ui.m_cmboxSampleTypeCH1->setCurrentIndex(3);
				break;
			case 0x0203:
				ui.m_cmboxSampleTypeCH1->setCurrentIndex(4);
				break;
			default:
				break;
			}
		}

	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}
void aioTestPage::slotOnReadCurrentCompCh1Ready(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

		if (unit.valueCount() == 1)
		{

			ui.m_spinBoxCurrentCompCH1->setValue(unit.value(0));
		}

	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}
void aioTestPage::slotOnReadValueCh1Ready(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

		if (unit.valueCount() == 1)
		{

			//ui.m_dataCh1-(unit.value(0));
			ui.m_dataCh1->setText(QString::number(unit.value(0)));
		}

	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}

void aioTestPage::slotOnWriteCh1Click(void)
{
	if (!m_master)
		return;
	if (m_master->state() == QModbusDevice::ConnectedState)
	{
		QModbusDataUnit writeUnitSampleType = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, SAMPLETYPE_CH1_REG_OFF, 1);
		switch (ui.m_cmboxSampleTypeCH1->currentIndex())
		{
		case 0:
			writeUnitSampleType.setValue(0, 0x0101);
			break;
		case 1:
			writeUnitSampleType.setValue(0, 0x0102);
			break;
		case 2:
			writeUnitSampleType.setValue(0, 0x0201);
			break;
		case 3:
			writeUnitSampleType.setValue(0, 0x0202);
			break;
		case 4:
			writeUnitSampleType.setValue(0, 0x0203);
			break;
		default:
			writeUnitSampleType.setValue(0, 0x0101);
			break;
		}

		if (auto *reply1 = m_master->sendWriteRequest(writeUnitSampleType, SLAVE_ID))
		{
			if (!reply1->isFinished())
			{
				connect(reply1, &QModbusReply::finished, this, &aioTestPage::slotOnWriteSampleTypeCh1Ready);
			}
			else
			{
				delete reply1; // broadcast replies return immediately
			}
		}

		QModbusDataUnit writeUnitCurrentComp = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, CURRENT_SAMPLE_COMP_CH1_OFF, 1);
		writeUnitCurrentComp.setValue(0, (quint16)ui.m_spinBoxCurrentCompCH1->value());
		if (auto *reply2 = m_master->sendWriteRequest(writeUnitCurrentComp, SLAVE_ID))
		{
			if (!reply2->isFinished())
			{
				connect(reply2, &QModbusReply::finished, this, &aioTestPage::slotOnWriteCurrentCompCh1Ready);
			}
			else
			{
				delete reply2; // broadcast replies return immediately
			}
		}

	}
}
void aioTestPage::slotOnWriteCh1Ready(void)
{

}
void aioTestPage::slotOnWriteSampleTypeCh1Ready(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

	}
	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}

	reply->deleteLater();
}
void aioTestPage::slotOnWriteCurrentCompCh1Ready(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

	}
	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}

	reply->deleteLater();
}

void aioTestPage::slotOnReadCh2Click(void)
{
	if (!m_master)
		return;
	if (m_master->state() == QModbusDevice::ConnectedState)
	{
		if (auto *reply = m_master->sendReadRequest(QModbusDataUnit(QModbusDataUnit::HoldingRegisters, SAMPLETYPE_CH2_REG_OFF, 1), SLAVE_ID))
		{
			if (!reply->isFinished())
			{
				connect(reply, SIGNAL(finished()), this, SLOT(slotOnReadSampleTypeCh2Ready()));
			}
			else
			{
				delete reply;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}

		if (auto *reply2 = m_master->sendReadRequest(QModbusDataUnit(QModbusDataUnit::HoldingRegisters, CURRENT_SAMPLE_COMP_CH2_OFF, 1), SLAVE_ID))
		{
			if (!reply2->isFinished())
			{
				connect(reply2, SIGNAL(finished()), this, SLOT(slotOnReadCurrentCompCh2Ready()));
			}
			else
			{
				delete reply2;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}

		if (auto *reply3 = m_master->sendReadRequest(QModbusDataUnit(QModbusDataUnit::HoldingRegisters, VALUE_CH2_REG_OFF, 1), SLAVE_ID))
		{
			if (!reply3->isFinished())
			{
				connect(reply3, SIGNAL(finished()), this, SLOT(slotOnReadValueCh2Ready()));
			}
			else
			{
				delete reply3;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}
	}
}
void aioTestPage::slotOnReadCh2Ready(void)
{

}

void aioTestPage::slotOnReadSampleTypeCh2Ready(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

		if (unit.valueCount() == 1)
		{

			switch (unit.value(0))
			{
			case 0x0101:
				ui.m_cmboxSampleTypeCH2->setCurrentIndex(0);
				break;
			case 0x0102:
				ui.m_cmboxSampleTypeCH2->setCurrentIndex(1);
				break;
			case 0x0201:
				ui.m_cmboxSampleTypeCH2->setCurrentIndex(2);
				break;
			case 0x0202:
				ui.m_cmboxSampleTypeCH2->setCurrentIndex(3);
				break;
			case 0x0203:
				ui.m_cmboxSampleTypeCH2->setCurrentIndex(4);
				break;
			default:
				break;
			}
		}

	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}
void aioTestPage::slotOnReadCurrentCompCh2Ready(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

		if (unit.valueCount() == 1)
		{

			ui.m_spinBoxCurrentCompCH2->setValue(unit.value(0));
		}

	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}
void aioTestPage::slotOnReadValueCh2Ready(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

		if (unit.valueCount() == 1)
		{

			//ui.m_dataCh1-(unit.value(0));
			ui.m_dataCh2->setText(QString::number(unit.value(0)));
		}

	}

	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}

void aioTestPage::slotOnWriteCh2Click(void)
{
	if (!m_master)
		return;
	if (m_master->state() == QModbusDevice::ConnectedState)
	{
		QModbusDataUnit writeUnitSampleType = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, SAMPLETYPE_CH2_REG_OFF, 1);
		switch (ui.m_cmboxSampleTypeCH2->currentIndex())
		{
		case 0:
			writeUnitSampleType.setValue(0, 0x0101);
			break;
		case 1:
			writeUnitSampleType.setValue(0, 0x0102);
			break;
		case 2:
			writeUnitSampleType.setValue(0, 0x0201);
			break;
		case 3:
			writeUnitSampleType.setValue(0, 0x0202);
			break;
		case 4:
			writeUnitSampleType.setValue(0, 0x0203);
			break;
		default:
			writeUnitSampleType.setValue(0, 0x0101);
			break;
		}

		if (auto *reply1 = m_master->sendWriteRequest(writeUnitSampleType, SLAVE_ID))
		{
			if (!reply1->isFinished())
			{
				connect(reply1, &QModbusReply::finished, this, &aioTestPage::slotOnWriteSampleTypeCh2Ready);
			}
			else
			{
				delete reply1; // broadcast replies return immediately
			}
		}

		QModbusDataUnit writeUnitCurrentComp = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, CURRENT_SAMPLE_COMP_CH2_OFF, 1);
		writeUnitCurrentComp.setValue(0, (quint16)ui.m_spinBoxCurrentCompCH2->value());
		if (auto *reply2 = m_master->sendWriteRequest(writeUnitCurrentComp, SLAVE_ID))
		{
			if (!reply2->isFinished())
			{
				connect(reply2, &QModbusReply::finished, this, &aioTestPage::slotOnWriteCurrentCompCh2Ready);
			}
			else
			{
				delete reply2; // broadcast replies return immediately
			}
		}
	}
}
void aioTestPage::slotOnWriteCh2Ready(void)
{

}
void aioTestPage::slotOnWriteSampleTypeCh2Ready(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

	}
	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}

	reply->deleteLater();
}
void aioTestPage::slotOnWriteCurrentCompCh2Ready(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();

	}
	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}

	reply->deleteLater();
}

void aioTestPage::slotOnMasterUpdate(void)
{
	if (!m_master)
		return;
	if (m_master->state() == QModbusDevice::ConnectedState)
	{
		QModbusDataUnit readUnitCh1Value = QModbusDataUnit(QModbusDataUnit::HoldingRegisters, VALUE_CH1_REG_OFF, 2);
		if (auto *reply = m_master->sendReadRequest(readUnitCh1Value, SLAVE_ID))
		{
			if (!reply->isFinished())
			{
				connect(reply, SIGNAL(finished()), this, SLOT(slotonMasterUpdateReady()));
			}
			else
			{
				delete reply;
			}
		}
		else
		{
			QMessageBox::warning(NULL, tr("Error"), tr("Read error: ") + m_master->errorString(), QMessageBox::Yes);
		}

	}

}

void aioTestPage::slotonMasterUpdateReady(void)
{
	auto reply = qobject_cast<QModbusReply *>(sender());
	if (!reply)
		return;

	if (reply->error() == QModbusDevice::NoError)
	{
		const QModbusDataUnit unit = reply->result();
		
		if (unit.valueCount() == 2)
		{
			ui.m_dataCh1->setText(QString::number(unit.value(0)));
			ui.m_dataCh2->setText(QString::number(unit.value(1)));

		}

	}
	else if (reply->error() == QModbusDevice::ProtocolError)
	{

	}
	else
	{

	}
	reply->deleteLater();
}