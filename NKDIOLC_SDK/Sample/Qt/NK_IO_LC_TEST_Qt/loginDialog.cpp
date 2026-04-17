#include <qcoreapplication.h>
#include "loginDialog.h"

loginDialog::loginDialog(QWidget *parent)
	: QDialog(parent)
{
	ui.setupUi(this);

	m_pSettings = new QSettings(QCoreApplication::applicationDirPath() + "/config.ini", QSettings::IniFormat);
	QStringList sectionList = m_pSettings->childGroups();

	ui.comboBox->addItems(sectionList);
	QSettings set(QCoreApplication::applicationDirPath() + "/select.ini", QSettings::IniFormat);
	QString selectItem = set.value("/SELECTED/Name").toString();
	ui.comboBox->setCurrentText(selectItem);

	connect(ui.m_btnExit, &QPushButton::clicked, this, &loginDialog::slotOnExit);
	connect(ui.m_btnAccept, &QPushButton::clicked, this, &loginDialog::slotOnAccept);

}

loginDialog::~loginDialog()
{
}


void loginDialog::slotOnExit(void)
{
	delete m_pSettings;
	close();
}

void loginDialog::slotOnAccept(void)
{
	QString selectConfig = m_pSettings->value(ui.comboBox->currentText() + "/ConfigPath").toString();

	QSettings set(QCoreApplication::applicationDirPath() + "/select.ini", QSettings::IniFormat);
	set.setValue("/SELECTED/Name", ui.comboBox->currentText());
	set.setValue("/SELECTED/ConfigPath", selectConfig);
	delete m_pSettings;
	accept();
}