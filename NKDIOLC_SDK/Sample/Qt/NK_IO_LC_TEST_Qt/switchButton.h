#pragma once
#include <qwidget.h>
class QTimer;
class switchButton : public QWidget
{
	Q_OBJECT
public:
	enum ButtonStyle
	{
		ButtonStyleRect = 0,
		ButtonStyleCircleIn = 1,
		ButtonStyleCircleOut = 2,
		ButtonStyleImage = 3
	};

	switchButton(QWidget *parent = Q_NULLPTR);
	~switchButton();

protected:
	void mousePressEvent(QMouseEvent *);
	void resizeEvent(QResizeEvent *);
	void paintEvent(QPaintEvent *);
	void drawBg(QPainter *painter);
	void drawSlider(QPainter *painter);
	void drawText(QPainter *painter);
	void drawImage(QPainter *painter);

private:
	bool m_checked;               //checked
	ButtonStyle buttonStyle;    //Style

	QColor bgColorOff;          //background color when off
	QColor bgColorOn;           //background color when on

	QColor sliderColorOff;      //slider color when off
	QColor sliderColorOn;       //slider color when on

	QColor textColorOff;        //text color when off
	QColor textColorOn;         //text color when on

	QString textOff;            //text when off
	QString textOn;             //text when on

	QString imageOff;           //image when off
	QString imageOn;            //image when on

	int space;                  //slider space
	int rectRadius;             //

	int step;                   //step moved each time
	int startX;                 //slider start on axis X

	int endX;                   //slider end on axis X

	QTimer *timer;              //timer on repaint

private slots:
	void updateValue();

public:
	bool getChecked()const
	{
		return m_checked;
	}

	ButtonStyle getButtonStyle()const
	{
		return buttonStyle;
	}

	QColor getBgColorOff()const
	{
		return bgColorOff;
	}

	QColor getBgColorOn()const
	{
		return bgColorOn;
	}

	QColor getSliderColorOff()const
	{
		return sliderColorOff;
	}

	QColor getSliderColorOn()const
	{
		return sliderColorOn;
	}

	QColor getTextColorOff()const
	{
		return textColorOff;
	}

	QColor getTextColorOn()const
	{
		return textColorOn;
	}

	QString getTextOff()const
	{
		return textOff;
	}

	QString getTextOn()const
	{
		return textOn;
	}


	QString getImageOff()const
	{
		return imageOff;
	}

	QString getImageOn()const
	{
		return imageOn;
	}



	int getSpace()const
	{
		return space;
	}

	int getRectRadius()const
	{
		return rectRadius;
	}



public slots:

	//set checked 
	void setChecked(bool checked);

	//set style
	void setButtonStyle(ButtonStyle buttonStyle);


	//set backgroundcolor
	void setBgColor(QColor bgColorOff, QColor bgColorOn);

	//set slider color
	void setSliderColor(QColor sliderColorOff, QColor sliderColorOn);

	//set text color
	void setTextColor(QColor textColorOff, QColor textColorOn);

	//set text
	void setText(QString textOff, QString textOn);

	//set image
	void setImage(QString imageOff, QString imageOn);



	//set space
	void setSpace(int space);

	//set rect radius
	void setRectRadius(int rectRadius);

signals:

	void checkedChanged(bool checked);

};

