#include "navigationWidget.h"
#include <QPainter>
#include <QDebug>

navigationWidget::navigationWidget(QWidget *parent)
	: QWidget(parent)
{
	//ui.setupUi(this);
	backgroundColor = "#E4E4E4";
	selectedColor = "#2CA7F8";
	rowHeight = 40;
	currentIndex = 0;

	setMouseTracking(true);
	setFixedWidth(160);
}

navigationWidget::~navigationWidget()
{
}

void navigationWidget::addItem(const QString &title)
{
	listItems << title;

	update();
}

void navigationWidget::setWidth(const int &width)
{
	setFixedWidth(width);
}

void navigationWidget::setBackgroundColor(const QString &color)
{
	backgroundColor = color;

	update();
}

void navigationWidget::setSelectColor(const QString &color)
{
	selectedColor = color;

	update();
}

void navigationWidget::setRowHeight(const int &height)
{
	rowHeight = height;

	update();
}

void navigationWidget::paintEvent(QPaintEvent *)
{
	QPainter painter(this);
	painter.setRenderHint(QPainter::Antialiasing, true);

	// Draw background color.
	painter.setPen(Qt::NoPen);
	painter.setBrush(QColor(backgroundColor));
	painter.drawRect(rect());

	// Draw Items
	int count = 0;
	for (const QString &str : listItems) {
		QPainterPath itemPath;
		itemPath.addRect(QRect(0, count * rowHeight, width(), rowHeight));

		if (currentIndex == count) {
			painter.setPen("#FFFFFF");
			painter.fillPath(itemPath, QColor(selectedColor));
		}
		else {
			painter.setPen("#202020");
			painter.fillPath(itemPath, QColor(backgroundColor));
		}

		painter.drawText(QRect(10, count * rowHeight, width(), rowHeight), Qt::AlignVCenter | Qt::AlignLeft, str);

		++count;
	}
}

void navigationWidget::mouseMoveEvent(QMouseEvent *e)
{
	if (e->y() / rowHeight < listItems.count()) {
		// qDebug() << e->y() / rowHeight;
	}
}

void navigationWidget::mousePressEvent(QMouseEvent *e)
{
	if (e->y() / rowHeight < listItems.count()) {
		currentIndex = e->y() / rowHeight;

		emit currentItemChanged(currentIndex);

		update();
	}
}

void navigationWidget::mouseReleaseEvent(QMouseEvent *e)
{

}