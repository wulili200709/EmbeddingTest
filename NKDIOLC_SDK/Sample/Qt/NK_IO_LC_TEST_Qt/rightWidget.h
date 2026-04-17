#pragma once

#include <QWidget>
#include <QStackedLayout>
#include "lightControlPage.h"
#include "dioTestPage.h"
#include "aboutPage.h"


//#include "ui_rightWidget.h"

class rightWidget : public QWidget
{
	Q_OBJECT

public:
	rightWidget(CDioLcAdapter *adapter, QWidget *parent = Q_NULLPTR);
	~rightWidget();
	QStackedLayout *m_layout;

protected:
	void paintEvent(QPaintEvent *);
private:
	//Ui::rightWidget ui;

	lightControlPage *m_lightControlPage;
	dioTestPage *m_dioTestPage;
	aboutPage *m_aboutPage;

	CDioLcAdapter *m_adapter;
	
};
