#include "ledWidget.h"
#include <qpainter.h>

ledWidget::ledWidget(QWidget *parent)
	: QWidget(parent)
{
	m_darkerFactor = 300;
	m_onColor = Qt::green;
	m_offColor = Qt::gray;
	m_isOn = true;
}

ledWidget::~ledWidget()
{
}

QColor ledWidget::onColor() const
{
	return m_onColor;
}
QColor ledWidget::offColor() const
{
	return m_offColor;
}
QSize ledWidget::sizeHint() const
{
	return QSize(20, 20);
}
QSize ledWidget::minimumSizeHint() const
{
	return QSize(16, 16);
}

// Slot
void ledWidget::setOnColor(const QColor &color)
{
	if (m_onColor == color)
	{
		return;
	}
	m_onColor = color;
	update();
}

void ledWidget::setOffColor(const QColor &color)
{
	if (m_offColor == color)
	{
		return;
	}
	m_offColor = color;
	update();
}

void ledWidget::toggle()
{
	m_isOn = !m_isOn;
	update();
}
void ledWidget::turnOn(bool on )
{
	m_isOn = on;
	update();
}
void ledWidget::turnOff(bool off)
{
	//m_isOn = off;
	//update();
	turnOn(!off);
}

// Event
void ledWidget::paintEvent(QPaintEvent *event)
{
	int width = ledWidth();
	QPainter painter(this);
	painter.setRenderHint(QPainter::Antialiasing);
	QColor color = m_isOn ? m_onColor : m_offColor;

	QBrush brush;
	brush.setStyle(Qt::SolidPattern);
	brush.setColor(color);
	painter.setBrush(brush);

	painter.drawEllipse(1, 1, width - 1, width - 1);

	QPen pen;
	pen.setWidth(2);
	int pos = width / 5 + 1;
	int lightWidth = width * 2 / 3;
	int lightQuote = 130 * 2 / (lightWidth ? lightWidth : 1) + 100;

	while (lightWidth)
	{
		color = color.lighter(lightQuote);
		pen.setColor(color);
		painter.setPen(pen);
		painter.drawEllipse(pos, pos, lightWidth, lightWidth);
		lightWidth--;

		if (!lightWidth)
			break;

		painter.drawEllipse(pos, pos, lightWidth, lightWidth);
		lightWidth--;

		if (!lightWidth)
			break;

		painter.drawEllipse(pos, pos, lightWidth, lightWidth);
		pos++;
		lightWidth--;
	}

	//draw border
	painter.setBrush(Qt::NoBrush);

	int angle = -720;
	color = palette().color(QPalette::Light);

	for (int arc = 120; arc < 2880; arc += 240) {
		pen.setColor(color);
		painter.setPen(pen);
		int w = width - pen.width() / 2;
		painter.drawArc(pen.width() / 2, pen.width() / 2, w, w, angle + arc, 240);
		painter.drawArc(pen.width() / 2, pen.width() / 2, w, w, angle - arc, 240);
		color = color.darker(110);
	}
}
int ledWidget::ledWidth()const
{
	int width = qMin(this->width(), this->height());
	width -= 2;
	return width > 0 ? width : 0;
}

