
#include <Windows.h>
#include "mainWindow.h"
#include <QtWidgets/QApplication>
#include <qdesktopwidget.h>
#include <QLibrary>
#include <qtranslator.h>
#include <qfile.h>
#include <QDebug>
#include <tchar.h>
#include "loginDialog.h"

bool checkOne()
{
	//  Create Mutex
	HANDLE m_hMutex = CreateMutex(NULL, FALSE, _T("NKDIO_TEST_TOOL"));
	//  Check the errorCode
	if (GetLastError() == ERROR_ALREADY_EXISTS) {
		//  if the error code is exist, then close handle and reset the Mutex
		CloseHandle(m_hMutex);
		m_hMutex = NULL;
		//  exit
		return  false;
	}
	else
		return true;
}

void initUiByLanguage(const QString strLanguage)
{
	if (strLanguage.isEmpty())
	{
		return;
	}
	QString strLanguageFile;
	if (strLanguage.compare("en") == 0)
	{
		//strLanguageFile = qApp->applicationDirPath() + QString("/languages/%1/%2").arg(LHT_SYNCCLIENT_VERSION_PRODOCUTNAME).arg(LHT_SYNCCLIENT_EN_FILE);
		strLanguageFile = QString("nkio_test_tool_en.qm");
	}
	else if (strLanguage.compare("zh") == 0)
	{
		//strLanguageFile = qApp->applicationDirPath() + QString("/languages/%1/%2").arg(LHT_SYNCCLIENT_VERSION_PRODOCUTNAME).arg(LHT_SYNCCLIENT_ZH_FILE);
		strLanguageFile = QString("nkio_test_tool_zh.qm");
	}
	if (QFile(strLanguageFile).exists())
	{
		QTranslator *m_translator = new QTranslator;
		m_translator->load(strLanguageFile);
		qApp->installTranslator(m_translator);
	}
	else
	{
		qDebug() << "[houqd] authclient language file does not exists ...";
	}

}

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	if (!checkOne())
	{
		return 0;
	}
	initUiByLanguage("zh");
	qApp->setStyleSheet("QTabWidget{background-color: gray}");

	loginDialog *pLoginDialog = new loginDialog();
	if (pLoginDialog->exec() == QDialog::Accepted)
	{
		mainWindow *w = new mainWindow();
		w->setFixedSize(1024, 768);
		w->move((QApplication::desktop()->width() - w->width()) / 2,
			(QApplication::desktop()->height() - w->height()) / 2);
		w->setWindowIcon(QIcon(":/resources/logo.svg"));
		w->show();

		return a.exec();
	}
	else
	{
		delete pLoginDialog;
		return 0;
	}
	
}
