#pragma once

#include <QDialog>
#include <qsettings.h>
#include "ui_loginDialog.h"

class loginDialog : public QDialog
{
	Q_OBJECT

public:
	loginDialog(QWidget *parent = Q_NULLPTR);
	~loginDialog();

public slots:
	void slotOnExit(void);
	void slotOnAccept(void);
private:
	Ui::loginDialog ui;
	QSettings *m_pSettings;
};
