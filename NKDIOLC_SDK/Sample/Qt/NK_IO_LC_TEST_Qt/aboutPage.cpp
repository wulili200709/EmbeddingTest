#include "aboutPage.h"

aboutPage::aboutPage(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	QPixmap *pixmap = new QPixmap(":/logo/NodkaLogo");
	int width = ui.label->width();
	int height = ui.label->height();
	//QPixmap fitpixmap = pixmap->scaled(width,height,Qt::IgnoreAspectRatio,Qt::SmoothTransformation);
	QPixmap fitpixmap = pixmap->scaled(width*8, height* 6, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
	//pixmap->scaled(ui.label->size(), Qt::KeepAspectRatio);
	//ui.label->setScaledContents(true);
	ui.label->setPixmap(fitpixmap);
}

aboutPage::~aboutPage()
{
}
