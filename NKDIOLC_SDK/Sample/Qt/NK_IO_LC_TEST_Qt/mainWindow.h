#pragma once

#include <QtWidgets/QMainWindow>
#include <qstatusbar.h>
#include <qboxlayout.h>
#include <qlabel.h>
#include "navigationWidget.h"
#include "rightWidget.h"
#include "ui_mainWindow.h"
//#include "NKLCAdapter.h"

class mainWindow : public QMainWindow
{
    Q_OBJECT

public:
    mainWindow(QWidget *parent = Q_NULLPTR);

private:
    //Ui::mainWindowClass ui;
	QStatusBar *m_statusBar;
	QWidget *m_mainWidget;
	QHBoxLayout *m_mainLayout;
	//QVBoxLayout *rightLayout;
	navigationWidget *m_navigationWidget;
	rightWidget *m_rightWidget;

	CDioLcAdapter *m_adapter;
};
