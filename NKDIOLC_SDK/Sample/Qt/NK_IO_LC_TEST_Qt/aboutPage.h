#pragma once

#include <QWidget>
#include "ui_aboutPage.h"

class aboutPage : public QWidget
{
	Q_OBJECT

public:
	aboutPage(QWidget *parent = Q_NULLPTR);
	~aboutPage();

private:
	Ui::aboutPage ui;
};
