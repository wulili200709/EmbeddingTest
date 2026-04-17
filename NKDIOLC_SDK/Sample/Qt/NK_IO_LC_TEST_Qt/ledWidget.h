#pragma once

#include <QWidget>

class ledWidget : public QWidget
{
	Q_OBJECT

public:
	ledWidget(QWidget *parent = Q_NULLPTR);
	~ledWidget();

	QColor onColor() const;
	QColor offColor() const;
	QSize sizeHint() const;
	QSize minimumSizeHint() const;

public slots:
	void setOnColor(const QColor &color);
	void setOffColor(const QColor &color);

	void toggle();
	void turnOn(bool on = true);
	void turnOff(bool off = true);

protected:
	void paintEvent(QPaintEvent *event);
	int ledWidth()const;

private:
	int m_darkerFactor;
	QColor m_onColor;
	QColor m_offColor;
	bool m_isOn;


};
