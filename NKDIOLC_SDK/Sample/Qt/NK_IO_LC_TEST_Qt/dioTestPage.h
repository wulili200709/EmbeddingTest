#pragma once

#include <windows.h>
#include <QWidget>
#include <qlabel.h>
#include "ui_dioTestPage.h"
#include "switchButton.h"
#include "ledWidget.h"
#include "DioLcAdapter.h"
#include <qDebug>

class dioTestPage : public QWidget
{
	Q_OBJECT

public:
	dioTestPage(CDioLcAdapter * adapter, QWidget *parent = Q_NULLPTR);
	~dioTestPage();


protected slots:
	void onTimeout();
	void onStartClicked();
	void onDOChecked(bool);
	void onDOAllChecked(bool);


private:

	quint16 getButtonStatus();
	void setLedStatus(quint16 status);
	void initSinglePointTest();
	void enableSinglePointTest(bool enable);
	void resetSinglePointTest();
	void processSinglePointTest();




private:
	Ui::dioTestPage ui;
	
	bool             m_bLoaded;

	unsigned short diValue;
	unsigned short doValue;

	unsigned int startFlag;
	QTimer *timer;

	CDioLcAdapter *m_adapter;
	
	unsigned int m_loopTestSteps;
	unsigned short m_loopDiValue;
	unsigned short m_loopDoValue;
	unsigned short m_loopCount;
	int m_loopCountMax;

};
