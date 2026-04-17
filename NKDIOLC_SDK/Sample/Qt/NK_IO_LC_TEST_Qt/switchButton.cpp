#include "switchButton.h"
#include <qpainter.h>
#include <qevent.h>
#include <qtimer.h>
#include <qdebug.h>

switchButton::switchButton(QWidget *parent):QWidget(parent)
{
	m_checked = false;

	buttonStyle = ButtonStyleRect;

	bgColorOff = QColor(225, 225, 225);
	bgColorOn = QColor(225, 225, 225);

	sliderColorOff = QColor(100, 100, 100);
	sliderColorOn = QColor(100, 184, 255);

	textColorOff = QColor(255, 255, 255);
	textColorOn = QColor(10, 10, 10);

	textOff = "";
	textOn = "";

	imageOff = ":/button/btncheckoff2";
	imageOn = ":/button/btncheckon2";

	space = 2;
	rectRadius = 5;

	step = width() / 50;
	startX = 0;
	endX = 0;

	timer = new QTimer(this);
	timer->setInterval(10);

	//connect(timer, SIGNAL(timeout()), this, SLOT(updateValue()));

	setFont(QFont("Microsoft Yahei", 10));
}

switchButton::~switchButton()
{

}

void switchButton::mousePressEvent(QMouseEvent *)
{
	m_checked = !m_checked;
	emit checkedChanged(m_checked);

	//each time step moved to be 1/50
	step = width() / 50;

	// end point will be calculated automatically when after state changed
	if (m_checked)
	{
		if (buttonStyle == ButtonStyleRect) 
		{
			endX = width() - width() / 2;
		}
		else if (buttonStyle == ButtonStyleCircleIn)
		{
			endX = width() - height();
		}
		else if (buttonStyle == ButtonStyleCircleOut) 
		{
			endX = width() - height() + space;
		}
	}
	else 
	{
		endX = 0;
	}
	startX = endX;
	timer->start();
	updateValue();
}
void switchButton::resizeEvent(QResizeEvent *)
{
	step = width() / 50;
	//
	if (m_checked)
	{
		if (buttonStyle == ButtonStyleRect) 
		{
			startX = width() - width() / 2;
		}
		else if (buttonStyle == ButtonStyleCircleIn) 
		{
			startX = width() - height();
		}
		else if (buttonStyle == ButtonStyleCircleOut) 
		{
			startX = width() - height() + space;
		}

	}
	else 
	{
		startX = 0;
	}
	update();
}

void switchButton::paintEvent(QPaintEvent *)
{
	QPainter painter(this);

	painter.setRenderHint(QPainter::Antialiasing);

	if (buttonStyle == ButtonStyleImage) 
	{
		drawImage(&painter);
	}
	else 
	{
		drawBg(&painter);
		drawSlider(&painter);
		drawText(&painter);
	}
}

void switchButton::drawBg(QPainter *painter)
{
	painter->save();
	painter->setPen(Qt::NoPen);

	if (!m_checked)
	{
		painter->setBrush(bgColorOff);
	}
	else 
	{
		painter->setBrush(bgColorOn);
	}

	if (buttonStyle == ButtonStyleRect) 
	{
		painter->drawRoundedRect(rect(), rectRadius, rectRadius);
	}
	else if (buttonStyle == ButtonStyleCircleIn) 
	{
		QRect rect(0, 0, width(), height());

		int radius = rect.height() / 2;
		int circleWidth = rect.height();

		QPainterPath path;

		path.moveTo(radius, rect.left());
		path.arcTo(QRectF(rect.left(), rect.top(), circleWidth, circleWidth), 90, 180);
		path.lineTo(rect.width() - radius, rect.height());
		path.arcTo(QRectF(rect.width() - rect.height(), rect.top(), circleWidth, circleWidth), 270, 180);
		path.lineTo(radius, rect.top());

		painter->drawPath(path);

	}
	else if (buttonStyle == ButtonStyleCircleOut) 
	{
		QRect rect(space, space, width() - space * 2, height() - space * 2);
		painter->drawRoundedRect(rect, rectRadius, rectRadius);
	}

	painter->restore();
}
void switchButton::drawSlider(QPainter *painter)
{
	painter->save();
	painter->setPen(Qt::NoPen);
	if (!m_checked)
	{
		painter->setBrush(sliderColorOff);
	}
	else 
	{
		painter->setBrush(sliderColorOn);
	}

	if (buttonStyle == ButtonStyleRect)
	{

		int sliderWidth = width() / 2 - space * 2;
		int sliderHeight = height() - space * 2;
		QRect sliderRect(startX + space, space, sliderWidth, sliderHeight);
		painter->drawRoundedRect(sliderRect, rectRadius, rectRadius);

	}
	else if (buttonStyle == ButtonStyleCircleIn) 
	{
		QRect rect(0, 0, width(), height());
		int sliderWidth = rect.height() - space * 2;
		QRect sliderRect(startX + space, space, sliderWidth, sliderWidth);
		painter->drawEllipse(sliderRect);

	}
	else if (buttonStyle == ButtonStyleCircleOut)
	{
		QRect rect(0, 0, width() - space, height() - space);
		int sliderWidth = rect.height();
		QRect sliderRect(startX, space / 2, sliderWidth, sliderWidth);
		painter->drawEllipse(sliderRect);

	}
	painter->restore();
}
void switchButton::drawText(QPainter *painter)
{
	painter->save();

	if (!m_checked)
	{
		painter->setPen(textColorOff);
		painter->drawText(width() / 2, 0, width() / 2 - space, height(), Qt::AlignCenter, textOff);
	}
	else 
	{
		painter->setPen(textColorOn);
		painter->drawText(0, 0, width() / 2 + space * 2, height(), Qt::AlignCenter, textOn);
	}
	painter->restore();
}
void switchButton::drawImage(QPainter *painter)
{
	painter->save();
	QPixmap pix;
	if (!m_checked)
	{
		pix = QPixmap(imageOff);
	}
	else
	{
		pix = QPixmap(imageOn);
	}

	int targetWidth = pix.width();
	int targetHeight = pix.height();
	pix = pix.scaled(targetWidth, targetHeight, Qt::KeepAspectRatio, Qt::SmoothTransformation);
	int pixX = rect().center().x() - targetWidth / 2;
	int pixY = rect().center().y() - targetHeight / 2;
	QPoint point(pixX, pixY);
	painter->drawPixmap(point, pix);
	painter->restore();
}


void switchButton::updateValue()
{
	if (m_checked)
	{
		if (startX < endX)
		{
			startX = startX + step;
		}
		else 
		{
			startX = endX;
			timer->stop();
		}
	}
	else 
	{
		if (startX > endX) 
		{
			startX = startX - step;
		}
		else 
		{
			startX = endX;
			timer->stop();
		}
	}
	update();
}

// slots
//set checked 
void switchButton::setChecked(bool checked)
{
	if (this->m_checked != checked)
	{
		this->m_checked = checked;
		emit checkedChanged(m_checked);
		//each time step moved to be 1/50
		step = width() / 50;

		// end point will be calculated automatically when after state changed
		if (m_checked)
		{
			if (buttonStyle == ButtonStyleRect)
			{
				endX = width() - width() / 2;
			}
			else if (buttonStyle == ButtonStyleCircleIn)
			{
				endX = width() - height();
			}
			else if (buttonStyle == ButtonStyleCircleOut)
			{
				endX = width() - height() + space;
			}
		}
		else
		{
			endX = 0;
		}
		startX = endX;
		//mousePressEvent(0);
		//update();
		//timer->start();
		updateValue();
	}
}

//set style
void switchButton::setButtonStyle(ButtonStyle buttonStyle)
{
	this->buttonStyle = buttonStyle;
	update();
}


//set backgroundcolor
void switchButton::setBgColor(QColor bgColorOff, QColor bgColorOn)
{
	this->bgColorOff = bgColorOff;
	this->bgColorOn = bgColorOn;
	update();
}

//set slider color
void switchButton::setSliderColor(QColor sliderColorOff, QColor sliderColorOn)
{
	this->sliderColorOff = sliderColorOff;
	this->sliderColorOn = sliderColorOn;
	update();
}

//set text color
void switchButton::setTextColor(QColor textColorOff, QColor textColorOn)
{
	this->textColorOff = textColorOff;
	this->textColorOn = textColorOn;
	update();
}

//set text
void switchButton::setText(QString textOff, QString textOn)
{
	this->textOff = textOff;
	this->textOn = textOn;
	update();
}

//set image
void switchButton::setImage(QString imageOff, QString imageOn)
{
	this->imageOff = imageOff;
	this->imageOn = imageOn;
	update();
}

//set space
void switchButton::setSpace(int space)
{
	this->space = space;
	update();
}

//set rect radius
void switchButton::setRectRadius(int rectRadius)
{
	this->rectRadius = rectRadius;
	update();
}





