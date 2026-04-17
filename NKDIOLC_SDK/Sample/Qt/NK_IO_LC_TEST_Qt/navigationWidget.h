#pragma once

#include <QWidget>
#include <QMouseEvent>
//#include "ui_navigationWidget.h"

class navigationWidget : public QWidget
{
	Q_OBJECT

public:
	navigationWidget(QWidget *parent = Q_NULLPTR);
	~navigationWidget();

	void addItem(const QString &title);
	void setWidth(const int &width);
	void setBackgroundColor(const QString &color);
	void setSelectColor(const QString &color);
	void setRowHeight(const int &height);

protected:
	void paintEvent(QPaintEvent *);
	void mouseMoveEvent(QMouseEvent *);
	void mousePressEvent(QMouseEvent *);
	void mouseReleaseEvent(QMouseEvent *);

private:
	QList<QString> listItems;
	QString backgroundColor;
	QString selectedColor;
	int rowHeight;
	int currentIndex;

signals:
	void currentItemChanged(int index);
};
