#include "rightWidget.h"
#include <QPainter>

rightWidget::rightWidget(CDioLcAdapter *adapter, QWidget *parent)
	: QWidget(parent)
	, m_adapter(adapter)
{
	//ui.setupUi(this);
	m_layout = new QStackedLayout(this);
	//m_connectionPage = new connectionPage();
	//m_adapter = new CDioLcAdapter();
	if (m_adapter->m_bDioNum >= 1)
	{
		m_dioTestPage = new dioTestPage(m_adapter);
		m_layout->addWidget(m_dioTestPage);
	}
	//if (m_adapter->m_bLcEnabled)
	//{
		m_lightControlPage = new lightControlPage(m_adapter);
		m_layout->addWidget(m_lightControlPage);
	//}
	m_aboutPage = new aboutPage();
	m_layout->addWidget(m_aboutPage);

	m_layout->setCurrentIndex(0);
	
}

rightWidget::~rightWidget()
{
}


void rightWidget::paintEvent(QPaintEvent *)
{
	QPainter painter(this);

	painter.setPen(QPen(QColor("#FFFFFF")));
	painter.setBrush(QColor("#FFFFFF"));
	painter.drawRect(rect());
}