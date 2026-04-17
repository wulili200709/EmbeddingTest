#include "mainWindow.h"

mainWindow::mainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    //ui.setupUi(this);
	m_mainWidget = new QWidget;
	m_mainLayout = new QHBoxLayout(m_mainWidget);

	m_navigationWidget = new navigationWidget();
	m_navigationWidget->setRowHeight(50);

	m_adapter = new CDioLcAdapter();

	//m_navigationWidget->addItem(tr("Connect"));
	if (m_adapter->m_bDioNum >= 1)
	{
		m_navigationWidget->addItem(tr("DIO TEST"));
	}
	//if (m_adapter->m_bLcEnabled)
	//{
		m_navigationWidget->addItem(tr("LIGHT CONTROL"));
	//}
	m_navigationWidget->addItem(tr("ABOUT"));
	

	m_rightWidget = new rightWidget(m_adapter);
	m_rightWidget->setFixedWidth(1024 - m_navigationWidget->width());
	//m_rightWidget->setModbusMasterInstance(m_master);

	m_mainLayout->addWidget(m_navigationWidget);
	m_mainLayout->addWidget(m_rightWidget);
	m_mainWidget->setLayout(m_mainLayout);

	m_mainLayout->setContentsMargins(0, 0, 0, 0);

	m_statusBar = new QStatusBar(this);
	m_statusBar->setObjectName(QString::fromUtf8("statusBar"));
	setStatusBar(m_statusBar);

	setCentralWidget(m_mainWidget);
	//setWindowTitle(tr("Nodka DIO TEST TOOL"));
	setWindowTitle(tr("NK_TEST_TOOL"));

	connect(m_navigationWidget, &navigationWidget::currentItemChanged, this, [=](int index) {
		m_rightWidget->m_layout->setCurrentIndex(index);

	});
}
