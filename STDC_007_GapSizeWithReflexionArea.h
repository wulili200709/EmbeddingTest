/*******************************************************************************
**  Project		: BM0xxxxx                                                    **
********************************************************************************
** Filename		: xxx.h
** Programmrer	: M. Meier
** Date			: 15.10.2018
** Version		: 2.00 		(camera project version)
********************************************************************************
** Changelog:
** Date			Version	Programmer		Description
** -----------------------------------------------------------------------------
** 15.10.18		0.00	M. Meier		New english camera template created
********************************************************************************/
// Note: Collaps/close all regions: Ctrl+M+O/L

#pragma message( " - Compiling " __FILE__ )
#pragma warning( disable : 4005 ) // hide warning
#pragma warning( disable : 4068 ) // hide warning (unknown Pragma (region) VS2003)

// ToDo: Rename namespace name to the name of the inspection
using namespace nSTDC_007_GapSizeWithReflexionArea;

//-------------------------------	Functions	--------------------------------


/// <summary>
/// Procedure of image evaluation.
/// </summary>
void ImageEvaluationProcedure()
{
	//---------------------------  Begin of block check 1  ----------------------------------

#pragma region BlockCheck

	if (ShutterSpecialSettingUsed)
	{
		snprintf(str_de, sizeof(str_de), "Shutter erhöht %i", Vid[KamNr].Wert[5]);
		snprintf(str_en, sizeof(str_en), "Shutter increased %i", Vid[KamNr].Wert[5]);
		SelectLanguage(string1, str_de, str_en);

		CHalBase::PaintText(string1, 110, 20, WindowWithPaintings, 200, 200, 200, 14);
	}

	UsingShutterSpecialSetting = true;

	if (ZaehlerMessdurchlauf == 2)
	{
		PZaehler[KamNr][1].Wert++;
	}


	////////////////////////
	// 1.1 Article is missing
#pragma region
	InspectionNo			= 2;
	RejectStatisticGroupNo	= 1;
	RejectStatisticEntryNo	= 1;
	MeasurementChartNo		= 0;
	Paintings				= PaintingsBrightFields();
	if (!Reject() && InspectionActive(InspectionNo))
	{
		TimeMeasurementStart();

		// set tolerance
		MinValue = 5; // minimum deviation

		// set ROI
		X1 = TargetPositionX - MmToPixel(1.00);
		Y1 = TargetPositionY - MmToPixel(1.00);
		X2 = TargetPositionX + MmToPixel(1.00);
		Y2 = TargetPositionY + MmToPixel(1.00);

		// generate region and image
		Roi = HRegion::GenRectangle1(Y1, X1, Y2, X2);
		HImage hImg = poHImg->getImg()->ReduceDomain(Roi).Rgb1ToGray();
		
		if (Paintings)
		{
			CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 0, 0, LineWidth*2); 	// shadow
			CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth); 	// automatically red painted in case of reject
		}

		Roi.Intensity(hImg, &Value);

		// check result
		if (Value < MinValue)
		{
			// set reject message
			snprintf(str_de, sizeof(str_de), "Artikel fehlt, Streuung = %.2f (Min %.2f)", Value, MinValue);
			snprintf(str_en, sizeof(str_en), "Article is missing, deviation = %.2f (Min %.2f)", Value, MinValue);
			SelectLanguage(RejectMessage, str_de, str_en);

			// set reject
			RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectArticleIsMissing, Roi);
		}

		// set data chart
		MeasurementChartSet(MeasurementChartNo, Value, MinValue);

		TimeMeasurementStop();
	}
#pragma endregion
#pragma endregion


	//---------------------------  End of block check 1  ------------------------------------

	//----------------------------  Loop for all poles  -------------------------------------
#pragma region DO NOT EDIT IN HERE

	// call general features
	if (EndBlockCheck1())
		return; // in case of reject and !DontStopAfterFirstReject

	// Loop for all poles
	for (PoleNoInImage = 1; PoleNoInImage <= NoPolesInImage; PoleNoInImage++)
	{
		// call general features
		if (BeginPoleCheck())
			continue; // pole not started by PLC or reject and !DontStopAfterFirstReject

#pragma endregion
		//------------------------  Begin of pole check  ------------------------------------

		////////////////////////
		// 2.1 Search dom
#pragma region STDC_007-2_Pressure plate deformation
		DomCenterX = 0, DomCenterY = 0;
		hrDomArea.Reset();

		// Call additional dom search as non-standard 
		if (AdditionalDomeSearch)
			ImageEvaluationProcedure_AdditionalsDomSearch();

		InspectionNo			= 10;
		RejectStatisticGroupNo	= 2;
		RejectStatisticEntryNo	= 1;
		MeasurementChartNo		= 6;
		Paintings				= false;
		if (!RejectPole() && !AdditionalDomeSearch)
		{
			TimeMeasurementStart();

			// Suchbereich festlegen/einzeichnen, Bildausschnitt übernehmen
			HRegion SearchArea;
			if (SearchAreaAtTargetPosition)
				SearchArea = HRegion::GenRectangle2(PoleCenterY, PoleCenterX, 0, MmToPixel(TargetGapHeight * FaktorSearchArea_X), MmToPixel(TargetGapHeight * FaktorSearchArea_Y));
			else
				SearchArea = HRegion::GenRectangle1(DistanceImageBorder_Y, DistanceImageBorder_X, ImageHeight() - DistanceImageBorder_Y, KBildbreite[KamNr] - DistanceImageBorder_X);

			CHalBase::PaintRegion(SearchArea, WindowWithPaintings, 0, 0, 255);

			HImage	hiPImg = poHImg->getImg()->ReduceDomain(SearchArea);

			int MinSchwelle = SearchDomSN;
			if (ArticleParameter(AP_GapCoating) >= GapCoating_AU)
				MinSchwelle = SearchDomAU;

			// Kanten filtern
			HXLDContArray hxa = hiPImg.EdgesSubPix("shen",	// Gewünschter Kanten-Operator
								AlphaEdgesSubPixDom,		// Filterparameter: kleine Werte bewirken starke Glättung, also auch weniger Bilddetails
								MinSchwelle,				// Untere Schwelle für Hysterese-Schwellenwertoperation.
								HighEdgesSubPixDom);		// Obere Schwelle für Hysterese-Schwellenwertoperation

			//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 251, 251, 251);

			// Die Linien auf Länge abfragen
			hxa = hxa.SelectContoursXld("contour_length",					// Selektionsmerkmal
												MmToPixel(0.01),			// Untere Schranke
												9999,						// Obere Schranke.
												0.5,						// Untere Schranke (ohne Bedeutung)
												0.5);						// Obere Schranke (ohne Bedeutung)
			
			//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 251, 251, 251);

			bool error = false;
			HTuple row, column, radius;

			// Flecken im Dombereich löschen über den Durchmesser der Region
			if (Filtermode_DiameterPoints)
			{
				HXLDContArray hxaTemp;
				for (int i = 0; i < hxa.Num(); i++)
				{
					double rad;
					hxa[i].SmallestCircleXld(__nullptr, &rad);
					if (rad > MmToPixel(TargetGapHeight / 7.5))
						hxaTemp.Append(hxa[i]);
				}
				hxa = hxaTemp;
			}

			// Nur Kanten die in die richtige Richtung laufen nutzen.
			if (Filtermode_LineDirection)
			{
				HXLDContArray hxaTemp;
				if (hxa.Num() > 0)
				{
					for (int i = 0; i < hxa.Num(); i++)
					{
						double r1, r2, c1, c2;
						r1 = hxa[i].SmallestRectangle1Xld(&c1, &r2, &c2);

						if (Differenz(r1,r2) > Differenz(c1,c2) && GapDirection == GapDirections::Vertical)
							hxaTemp.Append(hxa[i]);

						if (Differenz(r1,r2) < Differenz(c1,c2) && GapDirection == GapDirections::Horizontal)
							hxaTemp.Append(hxa[i]);
					}
					hxa = hxaTemp;
				}
				else
					error = true;
			}

			// Flecken im Dombereich über die Position löschen
			if (Filtermode_PointPosition)
			{
				HXLDContArray hxaTemp;
				if (hxa.Num() > 0)
				{
					HTuple medianColumn, medianRow;
					row = hxa.SmallestCircleXld(&column, __nullptr);
					medianColumn = column.Median();
					medianRow = row.Median();

					//DumpTuple(row);
					//DumpTuple(column);
					//DumpTuple(medianRow);
					//DumpTuple(medianColumn);
					int gapHeight = MmToPixel(TargetGapHeight * 0.90);

					for (int i = 0; i < hxa.Num(); i++)
					{
						double distance = DistancePP(medianColumn[0].D(), medianRow[0].D(), column[i].D(), row[i].D());

						double r = row[i].D();
						double c = column[i].D();
						//CHalBase::PaintCross(column[i].D(), row[i].D(), 10, Anz, 0, 0, 255);
						if (distance < gapHeight)
							hxaTemp.Append(hxa[i]);
					}
				}
				else
					error = true;

				hxa = hxaTemp;

				//CHalBase::PaintCross(medianColumn[0].D(), medianRow[0].D(), 40, WindowWithPaintings, 0, 0, 255);
				//CHalBase::PaintCircle(medianColumn[0].D(), medianRow[0].D(), Pixelmass_X(KamNr, TargetGapHeight * 0.90) , WindowWithPaintings, 0, 250, 250);
				//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 151, 151, 151);
			}

			// Kanten voll umfänglich nutzen.
			if (Filtermode_SmallestCircleAllEdges)
			{
				if (hxa.Num() > 0)
				{
					HRegion hrRec;
					for (int i = 0; i < hxa.Num(); i++)
					{
						double r1, c1, r;
						r1 = hxa[i].SmallestCircleXld(&c1, &r);

						HRegion hr = HRegion::GenCircle(r1, c1, r);
						CHalBase::PaintRegion(hr, WindowWithPaintings, 0, 150 + i, 150 + i);
						hrRec = hrRec.Union2(hr);
					}
					double row1, col1, r1;
					row1 = hrRec.SmallestCircle(&col1, &r1);
					HRegion hr = HRegion::GenCircle(row1, col1, r1);
					//CHalBase::PaintRegion(hr, WindowWithPaintings, 0, 250, 250, 2);
					hxa[0] = hr.GenContourRegionXld("border");
				}
				else
					error = true;
			}

			// Domloch suchen
			if (hxa.Num() > 0)
			{
				HXLDContArray hxaUnion = UnionXld(hxa);
				row = hxaUnion.SmallestCircleXld(&column, &radius);
				
				PoleCenterY = (int)row[0].D();
				PoleCenterX = (int)column[0].D();
				DomCenterY = row[0].D();
				DomCenterX = column[0].D();

				hrDomArea = HRegion::GenCircle(row, column, radius);
				hrDomArea = hrDomArea.Difference(hrDomArea.Difference(SearchArea));
				CHalBase::PaintRegion(hrDomArea, WindowWithPaintings, 0, 0, 255);

				Value = PixelToMm(radius[0].D() * 2);
				MinValue = TargetGapHeight * 0.75;
				MaxValue = TargetGapHeight * 1.90;
			}
			else
				error = true;

			if (InspectionActive(InspectionNo) && (Value < MinValue || Value > MaxValue) || error)
			{
				// set reject message
				if (error)
				{
					snprintf(str_de, sizeof(str_de), "Keine Kanten im Domloch gefunden");
					snprintf(str_en, sizeof(str_en), "No edges in dom found");
				}
				else
				{
					snprintf(str_de, sizeof(str_de), "Domloch nicht korrekt gefunden: %.2f (Min: %.2f / Max: %.2f) (Gabel fehlt)", Value, MinValue, MaxValue);
					snprintf(str_en, sizeof(str_en), "Dom not right found: %.2f (Min: %.2f / Max: %.2f) (Gabel fehlt)", Value, MinValue, MaxValue);
				}

				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectObjectSearch);

				CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 251, 251, 251);
				CHalBase::PaintRegion(hrDomArea, WindowWithPaintings, 255, 0, 0, 3);
			}
			else
			{
				if (GapDirection == GapDirections::Horizontal)
				{
					CHalBase::PaintLine(DomCenterX - radius[0].D() - 20, DomCenterY, DomCenterX - radius[0].D() + 20, DomCenterY, WindowWithPaintings, 0, 0, 255);
					CHalBase::PaintLine(DomCenterX + radius[0].D() - 20, DomCenterY, DomCenterX + radius[0].D() + 20, DomCenterY, WindowWithPaintings, 0, 0, 255);
				}
				else
				{
					CHalBase::PaintLine(DomCenterX, DomCenterY - radius[0].D() - 20, DomCenterX, DomCenterY - radius[0].D() + 20, WindowWithPaintings, 0, 0, 255);
					CHalBase::PaintLine(DomCenterX, DomCenterY + radius[0].D() - 20, DomCenterX, DomCenterY + radius[0].D() + 20, WindowWithPaintings, 0, 0, 255);
				}
			}

			// set data chart
			MeasurementChartSet(MeasurementChartNo, Value, MinValue, MaxValue, MinValue - 0.5, MaxValue + 0.5);

			TimeMeasurementStop();
		}
#pragma endregion


		////////////////////////
		// 1.2 Gabelreflexion
		double RefGapBrightness = 0;
		if (UsingBackLight)
			RefGapBrightness = 200;

#pragma region
		InspectionNo = 3;
		RejectStatisticGroupNo = 1;
		RejectStatisticEntryNo = 2;
		MeasurementChartNo = 2;
		Paintings = false;
		if (!RejectPole() && !UsingBackLight)
		{
			TimeMeasurementStart();

			int phi = 0;
			if (GapDirection == GapDirections::Horizontal)
				phi = 90;
			HRegion hrRefGabelHelligkeit;
			hrRefGabelHelligkeit = HRegion::GenRectangle2(DomCenterY, DomCenterX,  DegToRad(phi), MmToPixel(GapSize_min / 4), MmToPixel(TargetGapHeight / 3));
			CHalBase::PaintRegion(hrRefGabelHelligkeit, WindowWithPaintings, 0, 0, 255);

			HImage	hiPImg = poHImg->getImg()->ReduceDomain(hrRefGabelHelligkeit);
			Hlong sizeRoi = hrRefGabelHelligkeit.Area();
			HRegion hrRefNotSaturated = hiPImg.Threshold(0.0, 254.0); // do not use pixels that are in brightness saturation
			Hlong sizeNotSaturated = hrRefNotSaturated.Area();
			if (sizeNotSaturated > sizeRoi * 0.5)
				hiPImg = poHImg->getImg()->ReduceDomain(hrRefNotSaturated);

			double deviation = 0;
			RefGapBrightness = hrRefGabelHelligkeit.Intensity(hiPImg, &deviation);

			sprintf(string1, "RefBrightness = %.0f", RefGapBrightness);
			CHalBase::PaintText(string1, PoleCenterX + 100, ImageHeight() - FontSize, WindowWithPaintings, 200, 200, 200, FontSize);

			MinValue = MinRefBrightness;
			Value = RefGapBrightness;

			// check result
			if (Value < MinValue && InspectionActive(InspectionNo))
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Gabelreflexion zu gering = %.0f (Min: %.0f)", Value, MinValue);
				snprintf(str_en, sizeof(str_en), "Gap reflexion too low = %.0f (Min: %.0f)", Value, MinValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectObjectSearch);
			}

			// set data chart
			MeasurementChartSet(MeasurementChartNo, Value, MinValue, Hidden);

			TimeMeasurementStop();
		}
#pragma endregion


		////////////////////////
		// x.x Filter Helligkeit spreizen
		HImage himgDom = poHImg->getImg()->ReduceDomain(hrDomArea);
		HImage himgDomFB = poHImg->getImg()->ReduceDomain(hrDomArea);
		bool FilterActiv = false;
#pragma region
		InspectionNo = 4;
		RejectStatisticGroupNo = -1;
		RejectStatisticEntryNo = -1;
		MeasurementChartNo = -1;
		Paintings = false;
		if (!RejectPole() && InspectionActive(InspectionNo) && !UsingBackLight)
		{
			TimeMeasurementStart();

			HImage	hiPImg = poHImg->getImg()->ReduceDomain(hrDomArea);

			double minGray, maxGray, Range;
			minGray = hrDomArea.MinMaxGray(hiPImg, 5.0, &maxGray, &Range);

			if (ArticleParameter(AP_GapCoating) >= GapCoating_AU && RefGapBrightness < 245)
			{
				himgDomFB = ScaleImageMinMax(hiPImg, minGray, maxGray);
				FilterActiv = true;
			}
			else if (ArticleParameter(AP_GapCoating) == GapCoating_SN && RefGapBrightness < 150)
			{
				himgDomFB = ScaleImageMinMax(hiPImg, minGray, maxGray);
				FilterActiv = true;
			}

			if (FilterActiv)
			{
				CHalBase::HImage3ToRgbh(himgDomFB, WindowAdditional2, CameraNo);
				snprintf(str_de, sizeof(str_de), "Filter aktiv");
				snprintf(str_en, sizeof(str_en), "Filter active");
				SelectLanguage(string1, str_de, str_en);
				CHalBase::PaintText(string1, 100, ImageHeight() - 20, WindowWithPaintings, 200, 200, 200, 14);
			}

			TimeMeasurementStop();
		}
#pragma endregion
		

		////////////////////////
		// 1.3 Bildschärfe
#pragma region
		InspectionNo = 5;
		RejectStatisticGroupNo = 1;
		RejectStatisticEntryNo = 3;
		MeasurementChartNo = 3;
		Paintings = false;
		if (!RejectPole() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();

			HImage sobelImage = himgDomFB.SobelAmp("sum_abs", 3);

			HRegion hrSobel;
			hrSobel = sobelImage.Threshold(MinSobelImageSharpness, 255);

			//CHalBase::HImage3ToRgbh(sobelImage, Zus1, KamNr);
			CHalBase::PaintRegionFilled(hrSobel, WindowAdditional1, 0, 0, 255);
			if (GapDirection == GapDirections::Horizontal)
			{
				hrSobel = hrSobel.OpeningRectangle1(10, 1);
				hrSobel = hrSobel.ClosingRectangle1(5, 1);
			}
			else
			{
				hrSobel = hrSobel.OpeningRectangle1(1, 10);
				hrSobel = hrSobel.ClosingRectangle1(1, 5);
			}
			CHalBase::PaintRegionFilled(hrSobel, WindowAdditional1, 255, 0, 0);
			HRegionArray hraSobel = hrSobel.Connection();

			if (GapDirection == GapDirections::Horizontal)
				hraSobel = hraSobel.SelectShape("width","and", Pixelmass_X(KamNr, TargetGapHeight /2.5), 99999);
			else
			hraSobel = hraSobel.SelectShape("height", "and", Pixelmass_X(KamNr, TargetGapHeight / 2.5), 99999);

			CHalBase::PaintRegionFilled(hraSobel, WindowAdditional1, 0, 255, 0);

			if (hraSobel.Num() > 0)
			{
				hrSobel = hraSobel.Union1();
				Value = hrSobel.Intensity(sobelImage, __nullptr);
			}
			else
				Value = 0;

			//sprintf(string1, "Schärfe = %.0f", Wert);
			//CHalBase::PaintText(string1, KBildbreite[KamNr] - 150, KBildhoehe[KamNr] - 30, Anz, 200, 200, 200, 20);

			MinValue = MinValueImageSharpness;
						

			// check result
			if (Value < MinValue)
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Bildschärfe nicht ausreichend = %.2f (Min %.2f)", Value, MinValue);
				snprintf(str_de, sizeof(str_de), "Image sharpness too low = %.2f (Min %.2f)", Value, MinValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectObjectSearch);

				CHalBase::PaintRegionFilled(hrSobel, WindowWithPaintings, 255, 0, 0);
				CHalBase::PaintRegion(hrDomArea, WindowWithPaintings, 255, 0, 0);
				snprintf(str_de, sizeof(str_de), "Bildschärfe zu gering");
				snprintf(str_en, sizeof(str_en), "Image sharpness too low");
				SelectLanguage(string1, str_de, str_en);
				CHalBase::PaintText(string1, DomCenterX, DomCenterY - 20, WindowWithPaintings, 255, 0, 0, FontSize);
			}

			// set data chart
			MeasurementChartSet(MeasurementChartNo, Value, MinValue, Hidden);

			TimeMeasurementStop();
		}
#pragma endregion


		////////////////////////
		//2.2 Gabelkanten
		HXLDContArray hxa;
		double RowBegin0, ColBegin0, RowEnd0, ColEnd0, RowBegin1, ColBegin1, RowEnd1, ColEnd1 = 0;
		double GapCenterY, GapCenterX, GapAngle, GapHeight, GapWidth = 0;
#pragma region STDC_007-2_Pressure plate deformation
		InspectionNo = -1;
		RejectStatisticGroupNo = 2;
		RejectStatisticEntryNo = 2;
		MeasurementChartNo = -1;
		Paintings = false;
		if (!RejectPole())
		{
			TimeMeasurementStart();

			double Alpha = 0;
			int Low = 0;
			int High = 0;

			// Hellere Fläche gibt die Toleranzen für die Suche vor.
			if (RefGapBrightness > BrightStep1)
			{
				Alpha = AlphaValueStep1;
				Low = LowValueStep1;
				High = HighValueStep1;
			}
			else if (RefGapBrightness > BrightStep2)
			{
				Alpha = AlphaValueStep2;
				Low = LowValueStep2;
				High = HighValueStep2;
			}
			else if (FilterActiv && RefGapBrightness > BrightStep3)
			{
				Alpha = AlphaValueStep3;
				Low = LowValueStep3;
				High = HighValueStep3;
			}
			else
			{
				Alpha = AlphaValueDefault;
				Low = LowValueDefault;
				High = HighValueDefault;
			}

			// Feste Werte vorgeben
			if (ConstValueGapSearch)
			{
				Alpha = AlphaValueConst;
				Low = LowValueConst;
				High = HighValueConst;
			}

			// Kanten filtern
			hxa = himgDomFB.ReduceDomain(hrDomArea).EdgesSubPix("shen",				// Gewünschter Kanten-Operator
				Alpha,																// Filterparameter: kleine Werte bewirken starke Glättung, also auch weniger Bilddetails
				Low,																// Untere Schwelle für Hysterese-Schwellenwertoperation.
				High);																// Obere Schwelle für Hysterese-Schwellenwertoperation.
			//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 0, 0, 0);

			// Krumme Linien in einzelne Segmente teilen
			hxa = hxa.SegmentContoursXld("lines",									// Modus für die Segmentation der Konturen.
				5,																	// Einzugsbereich für die Glättung der Konturen.
				2.0,																// Maximaler Abstand zwischen einer Kontur und der approximierenden Gerade (erster Durchlauf).
				3.0);																// Maximaler Abstand zwischen einer Kontur und der approximierenden Gerade (zweiter Durchlauf).
			//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 0, 0, 0, 1);

			// Die Linien auf Länge abfragen
			hxa = hxa.SelectContoursXld("contour_length",							// Selektionsmerkmal
				MmToPixel(0.10),													// Untere Schranke
				9999,																// Obere Schranke.
				0.5,																// Untere Schranke (ohne Bedeutung)
				0.5);																// Obere Schranke (ohne Bedeutung)
			//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 0, 0, 0, 1);

			// Linien auf einer Geraden verbinden
			hxa = hxa.UnionCollinearContoursExtXld(MmToPixel(TargetGapHeight / 3),	//Maximaler Abstand der Endpunkte in Richtung der Referenzregressionsgeraden.
				MaxDistanceCloseEdges,												//Maximaler Abstand der Endpunkte in Richtung der Referenzregressionsgeraden relativ zur Länge der zu verlängernden Kontur.
				MaxDistanceCloseEdges,												//Maximaler Abstand der Konturen von der Referenzregressionsgeraden.
				RAD(5),																//Maximale Richtungsdifferenz.
				10.0,																//Maximal erlaubter Überlappungsbereich.
				-1.0,																//Maximaler Regressionsfehler der resultierenden Konturen (NICHT BENUTZT).
				1.0,																//Schwellwert zum Begrenzen der Gesamtkosten der Vereinigung 
				1.0,																//Einfluss des Abstandes in Linienrichtung auf die Gesamtkosten.
				1.0,																//Einfluss des Abstandes von der Regressionsgeraden auf die Gesamtkosten.
				1.0,																//Einfluss der Winkeldifferenz auf die Gesamtkosten.
				1.0,																//Einfluss von Störungen der Linie durch das Verbindungsstück (Überlappung und Winkelabweichung) auf die Gesamtkosten.
				0.0,																//Einfluss von Regressionsfehlern der resultierenden Kontur auf die Gesamtkosten (NICHT BENUTZT).
				"attr_keep");														//Modus für die Behandlung der Konturattribute
			//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 0, 0, 0, 1);

			// Die Linien auf Länge abfragen
			hxa = hxa.SelectContoursXld("contour_length",							// Selektionsmerkmal
				MmToPixel(TargetGapHeight / 2.5),									// Untere Schranke
				9999,																// Obere Schranke.
				0.5,																// Untere Schranke (ohne Bedeutung)
				0.5);																// Obere Schranke (ohne Bedeutung)
			//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 0, 0, 0, 1);

			// Alle Kanten einzeichnen
			int count = hxa.Num();
			for (int i = 0; i < count; i++)
			{
				int color = 100;
				CHalBase::PaintXLDCont(hxa[i], WindowWithPaintings, color + i, color + i, color + i, 1);
				if (i + color > 250)
					color = 255;
			}

			// Krumme Linien verwerfen
			if (count > 0)
			{
				HTuple circ = hxa.CircularityXld();
				//DumpTuple(circ);
				hxa = hxa.SelectShapeXld("circularity", "and", 0, MaxCircularity);
				//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 50, 50, 50, 1);
				count = hxa.Num();
			}

			// Horizontale Linien verwerfen
			if (count > 0 && GapDirection == GapDirections::Vertical)
			{
				HXLDContArray hxaPhiN = hxa.SelectShapeXld("rect2_phi", "and", -2.0, -1.0);
				HXLDContArray hxaPhiP = hxa.SelectShapeXld("rect2_phi", "and", 1.0, 2.0);
				hxa = hxaPhiN.Append(hxaPhiP);
				//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 50, 50, 50, 1);
				count = hxa.Num();
			}
			else
			{
				hxa = hxa.SelectShapeXld("rect2_phi", "and", -1.0, 1.0);
			}


			// Zickzack Linien verwerfen
			if (count > 0)
			{
				hxa = hxa.SelectShapeXld("rect2_len2", "and", 0, 3.3);
				//CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 50, 50, 50, 1);
				count = hxa.Num();
			}

			// Die Konturen nach Länge sortieren
			HTuple row, column, phi, width, height;
			HTuple indexByHeight;
			row = hxa.SmallestRectangle2Xld(&column, &phi, &height, &width);
			indexByHeight = height.SortIndex().Inverse();
			//DumpTuple(indexByHeight);
			//DumpTuple(height);


			// VERTICAL
			if (GapDirection == GapDirections::Vertical)
			{
				// Auswertung für vertikal stehende Gabelkanten
				// Die beiden mittlersten Kanten werden genutzt
				// Je Domseite wird nur eine Kante (die längste) akzepiert
				count = hxa.Num();
				if (count >= 2 && InspectionActive(33))
				{
					HXLDContArray hxaTemp;
					HXLDContArray hxaTempLeft;
					HXLDContArray hxaTempRight;
					for (int i = 0; i < count; i++)
					{
						if (DomCenterX < column[i].I()) // rechts oder liks von Mitte?
							hxaTempRight.Append(hxa[i]);
						else
							hxaTempLeft.Append(hxa[i]);
					}

					HTuple columnRight;
					HTuple indexByColumn;
					if (hxaTempRight.Num() > 0)
					{
						hxaTempRight.SmallestRectangle2Xld(&columnRight, nullptr, nullptr, nullptr);
						indexByColumn = columnRight.SortIndex();
						//DumpTuple(indexByColumn);
						//DumpTuple(columnRight);
						hxaTemp.Append(hxaTempRight[indexByColumn[0].I()]);
					}
					if (hxaTempLeft.Num() > 0)
					{
						HTuple columnLeft;
						hxaTempLeft.SmallestRectangle2Xld(&columnLeft, nullptr, nullptr, nullptr);
						indexByColumn = columnLeft.SortIndex().Inverse();
						//DumpTuple(indexByColumn);
						//DumpTuple(columnLeft);

						//CHalBase::PaintXLDCont(hxaTempLeft[indexByColumn[0].I()], Anz, 0, 255,0);
						hxaTemp.Append(hxaTempLeft[indexByColumn[0].I()]);
					}
					hxa = hxaTemp;
				}

				// Auswertung für vertikal stehende Gabelkanten
				// Die beiden mittlersten Kanten werden genutzt
				// Je Domseite wird nur eine Kante (die längste) akzepiert
				// Wenn Sondermodus Polarität aktiv (PF34) ist ist die Position der Kanten egal.
				// Dann wird die Polarität der längsten Kante genommen und die längste Kante mit entgegengesetzter Polarität.
				count = hxa.Num();
				if (count >= 2 && !InspectionActive(33) && InspectionActive(34))
				{
					// Die beiden längsten Kanten werden genutzt
					HXLDContArray hxaTemp;
					HTuple Polaritaet = 0;
					int num = 2;
					for (int i = 0; i < num && i < count; i++)
					{
						// Die Erste (längste) Kontur hinzufügen
						if (i == 0)
						{
							hxaTemp.Append(hxa[indexByHeight[i].I()]);

							// Polarität messen. Ca. 1,5 für hell/dunkel oder ca. 4,5 für dunkel/hell
							Polaritaet = hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Mean();
							continue;
						}

						// Polarität messen. Ca. 1,5 für hell/dunkel oder ca. 4,5 für dunkel/hell
						if (InspectionActive(34)) // Abfrage nach Polarität
						{
							// Es wrd nur eine Kontur pro Domseite genommen.
							if (i > 0 && i < num)
							{
								//DumpTuple(Polaritaet);
								//DumpTuple(hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Mean());
								//CHalBase::PaintXLDCont(hxaTempRight[indexByColumn[0].I()], Anz, 0, 255,0);
								if (Polaritaet < 3) //Erste rechts?
								{
									if (hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Mean() > 3) //Aktuelle rechts?
										hxaTemp.Append(hxa[indexByHeight[i].I()]);
									else
										num++;
								}
								else
								{
									if (hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Mean() < 3) //Aktuelle links?
										hxaTemp.Append(hxa[indexByHeight[i].I()]);
									else
										num++;
								}
							}
						}
					}
					hxa = hxaTemp;
				}
			}

			// HORIZONTAL
			if (GapDirection == GapDirections::Horizontal)
			{
				// Auswertung für horizontal stehende Gabelkanten
				// Die beiden mittlersten Kanten werden genutzt
				// Je Domseite wird nur eine Kante (die längste) akzepiert
				count = hxa.Num();
				if (count >= 2 && InspectionActive(33))
				{
					HXLDContArray hxaTemp;
					HXLDContArray hxaTempTop;
					HXLDContArray hxaTempBottom;
					for (int i = 0; i < count; i++)
					{
						if (DomCenterY < row[i].I()) // rechts oder liks von Mitte?
							hxaTempBottom.Append(hxa[i]);
						else
							hxaTempTop.Append(hxa[i]);
					}

					HTuple indexByRow;
					if (hxaTempBottom.Num() > 0)
					{
						HTuple rowBottom = hxaTempBottom.SmallestRectangle2Xld(nullptr, nullptr, nullptr, nullptr);
						indexByRow = rowBottom.SortIndex();
						//DumpTuple(indexByRow);
						//DumpTuple(rowBottom);

						//CHalBase::PaintXLDCont(hxaTempBottom[indexByRow[0].I()], Anz, 0, 255,0);
						hxaTemp.Append(hxaTempBottom[indexByRow[0].I()]);
					}
					if (hxaTempTop.Num() > 0)
					{
						HTuple  rowTop = hxaTempTop.SmallestRectangle2Xld(nullptr, nullptr, nullptr, nullptr);
						indexByRow = rowTop.SortIndex().Inverse();
						//DumpTuple(indexByRow);
						//DumpTuple(rowTop);

						//CHalBase::PaintXLDCont(hxaTempLeft[indexByColumn[0].I()], Anz, 0, 255,0);
						hxaTemp.Append(hxaTempTop[indexByRow[0].I()]);
					}
					hxa = hxaTemp;
				}

				// Auswertung für horizontal stehende Gabelkanten
				// Die beiden mittlersten Kanten werden genutzt
				// Je Domseite wird nur eine Kante (die längste) akzepiert
				// Wenn Sondermodus Polarität aktiv (PF34) ist ist die Position der Kanten egal.
				// Dann wird die Polarität der längsten Kante genommen und die längste Kante mit entgegengesetzter Polarität.
				count = hxa.Num();
				if (count >= 2 && !InspectionActive(33) && InspectionActive(34))
				{
					// Die beiden längsten Kanten werden genutzt
					HXLDContArray hxaTemp;
					HTuple Polaritaet = 0;
					int num = 2;
					for (int i = 0; i < num && i < count; i++)
					{
						//DumpTuple(hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction"));

						// Die Erste (längste) Kontur hinzufügen
						if (i == 0)
						{
							hxaTemp.Append(hxa[indexByHeight[i].I()]);

							// Polarität messen. Ca. 0 oder 6,28 für hell/dunkel oder ca. 3,14 für dunkel/hell
							if (hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Min() > 1.5 && hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Max() < 4.5)
							Polaritaet = 3.14;
							continue;
						}

						// Polarität messen. Ca. 1,5 für hell/dunkel oder ca. 4,5 für dunkel/hell
						if (InspectionActive(34)) // Abfrage nach Polarität
						{
							// Es wrd nur eine Kontur pro Domseite genommen.
							if (i > 0 && i < num)
							{
								//DumpTuple(Polaritaet);
								//DumpTuple(hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Mean());
								CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 0, 255, 0, 1);
								if (Polaritaet == 3.14) //Erste rechts?
								{
									if (hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Min() < 1.50 || hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Max() > 4.50) //Aktuelle unten?
										hxaTemp.Append(hxa[indexByHeight[i].I()]);
									else
										num++;
								}
								else
								{
									if (hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Min() > 1.50 && hxa[indexByHeight[i].I()].GetContourAttribXld("edge_direction").Max() < 4.50) //Aktuelle oben?
										hxaTemp.Append(hxa[indexByHeight[i].I()]);
									else
										num++;
								}
							}
						}
					}
					hxa = hxaTemp;
				}
			}

			count = hxa.Num();
			if (count < 2)
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Zu wenig Gabelkanten gefunden");
				snprintf(str_en, sizeof(str_en), "Not enough gap edges found");
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectObjectSearch);

				CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 255, 0, 0);
			}

			// Die Beiden gefundenen Kanten fitten
			if (!RejectPole())
			{
				count = hxa.Num();
				if (count == 2)
				{
					RowBegin0 = hxa[0].FitLineContourXld("tukey", -1, 10, 5, 2.0, &ColBegin0, &RowEnd0, &ColEnd0, __nullptr, __nullptr, __nullptr);
					if (RowBegin0 > RowEnd0)
					{
						double x = ColBegin0;
						double y = RowBegin0;
						ColBegin0 = ColEnd0;
						RowBegin0 = RowEnd0;
						ColEnd0 = x;
						RowEnd0 = y;
					}
					if (UsingBackLight)
						CHalBase::PaintLine(ColBegin0, RowBegin0, ColEnd0, RowEnd0, WindowWithPaintings, 0, 200, 200, LineWidth + 1);
					else
						CHalBase::PaintLine(ColBegin0, RowBegin0, ColEnd0, RowEnd0, WindowWithPaintings, 200, 200, 200, LineWidth + 1);


					RowBegin1 = hxa[1].FitLineContourXld("tukey", -1, 10, 5, 2.0, &ColBegin1, &RowEnd1, &ColEnd1, __nullptr, __nullptr, __nullptr);
					if (RowBegin1 > RowEnd1)
					{
						double x = ColBegin1;
						double y = RowBegin1;
						ColBegin1 = ColEnd1;
						RowBegin1 = RowEnd1;
						ColEnd1 = x;
						RowEnd1 = y;
					}
					if (UsingBackLight)
						CHalBase::PaintLine(ColBegin1, RowBegin1, ColEnd1, RowEnd1, WindowWithPaintings, 0, 200, 200, LineWidth + 1);
					else
						CHalBase::PaintLine(ColBegin1, RowBegin1, ColEnd1, RowEnd1, WindowWithPaintings, 200, 200, 200, LineWidth + 1);

					// Gabelspalt und Mitte ermitteln
					HXLDCont hx = UnionXld(hxa);
					GapCenterY = hx.SmallestRectangle2Xld(&GapCenterX, &GapAngle, &GapHeight, &GapWidth);
					CHalBase::PaintCross(GapCenterX, GapCenterY, 10, WindowWithPaintings, 150, 150, 150);


					// Check if no edge at search area shape
					if (count == 2)
					{
						bool edgeOnShape = false;

						Hlong r1, r2, c1, c2;
						r1 = hrDomArea.SmallestRectangle1(&c1, &r2, &c2);

						double r, c, phi, w, h = 0;

						r = hxa[1].SmallestRectangle2Xld(&c, &phi, &h, &w);
						if (!edgeOnShape) edgeOnShape = Wert_in_Toleranz(c, c1, 2.0);
						if (!edgeOnShape) edgeOnShape = Wert_in_Toleranz(r, r1, 2.0);

						r = hxa[0].SmallestRectangle2Xld(&c, &phi, &h, &w);
						if (!edgeOnShape) edgeOnShape = Wert_in_Toleranz(c, c1, 2.0);
						if (!edgeOnShape) edgeOnShape = Wert_in_Toleranz(r, r1, 2.0);

						if (edgeOnShape)
						{
							// set reject message
							snprintf(str_de, sizeof(str_de), "Gabelfläche wird vom Suchbereich begrenzt");
							snprintf(str_en, sizeof(str_en), "Gap area are cutted by search area");
							SelectLanguage(RejectMessage, str_de, str_en);

							// set reject
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectObjectSearch);
							CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 255, 0, 0);
						}
					}
				}
				else
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Anzahl Gabelkanten nicht korrekt");
					snprintf(str_en, sizeof(str_en), "Count gap edges not okay");
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectObjectSearch);
					CHalBase::PaintXLDCont(hxa, WindowWithPaintings, 255, 0, 0);
				}
			}

			TimeMeasurementStop();
		}
#pragma endregion


		////////////////////////
		// 2.3 Kantenlänge
#pragma region STDC_007-2_Pressure plate deformation
		InspectionNo = 11;
		RejectStatisticGroupNo = 2;
		RejectStatisticEntryNo = 3;
		MeasurementChartNo = 7;
		Paintings = false;
		if (!RejectPole() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();

			// Messwert berechnen
			double r, c, phi, w, h = 0;
			r = hxa[1].SmallestRectangle2Xld(&c, &phi, &h, &w);
			double EdgeLengthL = h * 2;

			r = hxa[0].SmallestRectangle2Xld(&c, &phi, &h, &w);
			double EdgeLengthR = h * 2;

			MinValue = TargetGapHeight * 0.50;

			Value = PixelToMm(EdgeLengthR);
			if (!RejectPole() && Value < MinValue)
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Kantenlänge zu kurz = %.2fmm (Min: %.2fmm)", Value, MinValue);
				snprintf(str_en, sizeof(str_en), "Edge length too short = %.2fmm (Min: %.2fmm)", Value, MinValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectGapDamaged);

				CHalBase::PaintXLDCont(hxa[1], WindowWithPaintings, 250, 0, 0, LineWidth + 1);
			}

			// set data chart
			MeasurementChartSet(MeasurementChartNo, Value, MinValue, Hidden);

			Value = PixelToMm(EdgeLengthL);
			if (!RejectPole() && Value < MinValue)
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Kantenlänge zu kurz = %.2fmm (Min: %.2fmm)", Value, MinValue);
				snprintf(str_en, sizeof(str_en), "Edge length too short = %.2fmm (Min: %.2fmm)", Value, MinValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectGapDamaged);

				CHalBase::PaintXLDCont(hxa[0], WindowWithPaintings, 250, 0, 0, LineWidth + 1);
			}

			// set data chart
			MeasurementChartSet(MeasurementChartNo, Value, MinValue, Hidden);

			MaxValue = MaxDifferenceLengthEdges; // in % der Sollgabelhöhe

			// Die Längen Differenz zwischen den Gabelkanten darf nicht mehr als ...% der Sollgabelhöhe betragen
			Value = PixelToMm(Differenz(EdgeLengthL, EdgeLengthR));
			Value = Value / TargetGapHeight * 100;
			if (!RejectPole() && Value > MaxValue)
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Kantenlängen Differenz = %.0f%% (Max: %.0f%%)", Value, MaxValue);
				snprintf(str_en, sizeof(str_en), "Edge lenght difference = %.0f%% (Max: %.0f%%)", Value, MaxValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectGapDamaged);

				CHalBase::PaintXLDCont(hxa[0], WindowWithPaintings, 250, 0, 0, LineWidth + 1);
				CHalBase::PaintXLDCont(hxa[1], WindowWithPaintings, 250, 0, 0, LineWidth + 1);
			}

			// set data chart
			MeasurementChartSet(MeasurementChartNo + 1, Value, Hidden, MaxValue);

			TimeMeasurementStop();
		}
#pragma endregion


		////////////////////////
		// 2.4 Kantenwinkel
#pragma region STDC_007-2_Pressure plate deformation
		InspectionNo = 12;
		RejectStatisticGroupNo = 2;
		RejectStatisticEntryNo = 4;
		MeasurementChartNo = 9;
		Paintings = true;
		if (!RejectPole() && (InspectionActive(InspectionNo) || InspectionActive(InspectionNo + 1)))
		{
			TimeMeasurementStart();

			// Messwert berechnen
			double r, c, phi, w, h = 0;
			r = hxa[1].SmallestRectangle2Xld(&c, &phi, &w, &h);
			double AngleR = RadToDeg(phi);

			r = hxa[0].SmallestRectangle2Xld(&c, &phi, &w, &h);
			double AngleL = RadToDeg(phi);

			MaxValue = Beo_MaxEdgesAngle; // in °
			MaxValue = BeoSysGetValue(CritNr_STDC_007_EdgesAngle, SettNr_STDC_007_Edges_Angle, MaxValue);

			if (GapDirection == GapDirections::Horizontal)
				Value = Differenz(0, abs(AngleL));
			else
				Value = Differenz(90, abs(AngleL));

			if (!RejectPole() && Value > MaxValue && InspectionActive(InspectionNo))
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Gabelkanten zu schief = %.2f° (Max: %.2f°)", Value, MaxValue);
				snprintf(str_en, sizeof(str_en), "Gap edges angle wrong = %.2f° (Max: %.2f°)", Value, MaxValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectGapDamaged);

				CHalBase::PaintXLDCont(hxa[0], WindowWithPaintings, 250, 0, 0, LineWidth + 1);
			}

			// set data chart
			MeasurementChartSet(MeasurementChartNo, Value, Hidden, MaxValue);

			if (GapDirection == GapDirections::Horizontal)
				Value = Differenz(0, abs(AngleR));
			else
				Value = Differenz(90, abs(AngleR));

			if (!RejectPole() && Value > MaxValue && InspectionActive(InspectionNo))
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Gabelkanten zu schief = %.2f° (Max: %.2f°)", Value, MaxValue);
				snprintf(str_en, sizeof(str_en), "Gap edges angle wrong = %.2f° (Max: %.2f°)", Value, MaxValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectGapDamaged);

				CHalBase::PaintXLDCont(hxa[1], WindowWithPaintings, 250, 0, 0, LineWidth + 1);
			}

			// set data chart
			MeasurementChartSet(MeasurementChartNo + 1, Value, Hidden, MaxValue);

			MaxValue = Beo_MaxEdgesParallelism; // in ° laut Einzelteil Zeichnng
			MaxValue = BeoSysGetValue(CritNr_STDC_007_EdgesAngle, SettNr_STDC_007_Edges_Parallelism, MaxValue);

			if (AngleL < 0) AngleL += 180;
			if (AngleR < 0) AngleR += 180;
			double Anglediff = Differenz(AngleL, AngleR);
			if (Anglediff > 90 && Anglediff < 180) Anglediff = 180 - Anglediff;
			Value = Anglediff;
			if (!RejectPole() && Value > MaxValue && InspectionActive(InspectionNo + 1))
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Kanten nicht parallel = %.2f° (Max: %.2f°)", Value, MaxValue);
				snprintf(str_en, sizeof(str_en), "Edges not parallelism = %.2f° (Max: %.2f°)", Value, MaxValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectGapDamaged);

				CHalBase::PaintXLDCont(hxa[0], WindowWithPaintings, 250, 0, 0, 2);
				CHalBase::PaintXLDCont(hxa[1], WindowWithPaintings, 250, 0, 0, 2);
			}

			// set data chart
			MeasurementChartSet(MeasurementChartNo + 2, Value, Hidden, MaxValue);

			TimeMeasurementStop();
		}
#pragma endregion

		////////////////////////
		// 3.1 Gabelmass
#pragma region STDC_007-1_Gap size dimension
		InspectionNo = 17;
		RejectStatisticGroupNo = 3;
		RejectStatisticEntryNo = 1;
		MeasurementChartNo = 12;
		Paintings = true;
		if (!RejectPole() && (InspectionActive(InspectionNo) || InspectionActive(InspectionNo + 1)))
		{
			TimeMeasurementStart();

			if (CountGapMultiSize < 1)
				CountGapMultiSize = 1;

			// Toleranz des Gabelmaß
			MinValue = GapSize_min;
			MaxValue = GapSize_max;

			double transX1, transY1, transX2, transY2;
			HTuple hXGabelpos0, hYGabelpos0, hXGabelpos1, hYGabelpos1;
			double XGabelpos0, YGabelpos0, XGabelpos1, YGabelpos1;

			// Längere Kontur nutzen
			if (hxa[0].LengthXld() > hxa[1].LengthXld())
			{
				double xCen = (ColBegin0 + ColEnd0) / 2;
				int DiffMeasure = (RowEnd0 - RowBegin0) / CountGapMultiSize;
				int StartMeasure = RowBegin0 + DiffMeasure / 2;

				for (int i = 0; i < CountGapMultiSize; i++)
				{
					int y = StartMeasure + DiffMeasure * i;

					HTuple HomMat2D;
					hom_mat2d_identity(&HomMat2D); // Matrix erstellen
					hom_mat2d_rotate(HomMat2D, DegToRad(90), y, xCen, &HomMat2D); // Mittelpunkt und zu drehender Winkel des zu drehenden Objekts angeben
					affine_trans_pixel(HomMat2D, RowBegin0, ColBegin0, &transY1, &transX1); // Anfangspunkt der XLD Kontur drehen
					affine_trans_pixel(HomMat2D, RowEnd0, ColEnd0, &transY2, &transX2); // EndPunktpunkt der XLD Kontur drehen
					//CHalBase::PaintLine(transX1, transY1, transX2, transY2, WindowWithPaintings, 255, 0, 0);

					IntersectionLL(ColBegin0, RowBegin0, ColEnd0, RowEnd0, transX1, transY1, transX2, transY2, XGabelpos0, YGabelpos0);
					IntersectionLL(ColBegin1, RowBegin1, ColEnd1, RowEnd1, transX1, transY1, transX2, transY2, XGabelpos1, YGabelpos1);
					CHalBase::PaintCross(XGabelpos0, YGabelpos0, 10, WindowWithPaintings, 0, 0, 250, LineWidth);
					CHalBase::PaintCross(XGabelpos1, YGabelpos1, 10, WindowWithPaintings, 0, 0, 250, LineWidth);

					hXGabelpos0[i] = XGabelpos0;
					hYGabelpos0[i] = YGabelpos0;
					hXGabelpos1[i] = XGabelpos1;
					hYGabelpos1[i] = YGabelpos1;
				}
			}
			else
			{
				double xCen = (ColBegin1 + ColEnd1) / 2;
				
				int DiffMeasure = (RowEnd1 - RowBegin1) / CountGapMultiSize;
				int StartMeasure = RowBegin1 + DiffMeasure / 2;

				for (int i = 0; i < CountGapMultiSize; i++)
				{
					int y = StartMeasure + DiffMeasure * i;

					HTuple HomMat2D;
					hom_mat2d_identity(&HomMat2D); // Matrix erstellen
					hom_mat2d_rotate(HomMat2D, DegToRad(90), y, xCen, &HomMat2D); // Mittelpunkt und zu drehender Winkel des zu drehenden Objekts angeben
					affine_trans_pixel(HomMat2D, RowBegin1, ColBegin1, &transY1, &transX1); // Anfangspunkt der XLD Kontur drehen
					affine_trans_pixel(HomMat2D, RowEnd1, ColEnd1, &transY2, &transX2); // EndPunktpunkt der XLD Kontur drehen
					//CHalBase::PaintLine(transX1, transY1, transX2, transY2, WindowWithPaintings, 255, 0, 0);

					IntersectionLL(ColBegin0, RowBegin0, ColEnd0, RowEnd0, transX1, transY1, transX2, transY2, XGabelpos0, YGabelpos0);
					IntersectionLL(ColBegin1, RowBegin1, ColEnd1, RowEnd1, transX1, transY1, transX2, transY2, XGabelpos1, YGabelpos1);
					CHalBase::PaintCross(XGabelpos0, YGabelpos0, 10, WindowWithPaintings, 0, 0, 250, LineWidth);
					CHalBase::PaintCross(XGabelpos1, YGabelpos1, 10, WindowWithPaintings, 0, 0, 250, LineWidth);

					hXGabelpos0[i] = XGabelpos0;
					hYGabelpos0[i] = YGabelpos0;
					hXGabelpos1[i] = XGabelpos1;
					hYGabelpos1[i] = YGabelpos1;
				}
			}

			for (int i = 0; i < CountGapMultiSize; i++)
			{
				if (hXGabelpos0[i].ValType() == UndefVal || hXGabelpos0[i].ValType() == UndefVal || hXGabelpos0[i].ValType() == UndefVal || hXGabelpos0[i].ValType() == UndefVal)
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Fehler multi Gabelmass");
					snprintf(str_en, sizeof(str_en), "Error multi gap size Gabelmass");
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectObjectSearch);

					continue;
				}

				XGabelpos0 = hXGabelpos0[i].D();
				YGabelpos0 = hYGabelpos0[i].D();
				XGabelpos1 = hXGabelpos1[i].D();
				YGabelpos1 = hYGabelpos1[i].D();


				int PaintR = 50, PaintG = 200, PaintB = 50; // Standard: light green
				if (RefGapBrightness > 125)
				{
					PaintR = 0; PaintG = 150; PaintB = 0; // Bright areas: darker green
				}

				// Messwert berechnen
				Value = PixelToMm(DistancePP(XGabelpos0, YGabelpos0, XGabelpos1, YGabelpos1));

				if (InspectionActive(InspectionNo) && (Value < MinValue))
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Gabelmass zu klein = %.3fmm (Min: %.3fmm)", Value, MinValue);
					snprintf(str_en, sizeof(str_en), "Gap size too small = %.3fmm (Min %.3fmm)", Value, MinValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectGapSizeTooSmall);

					PaintR = 255; PaintG = 0; PaintB = 0;
					CHalBase::PaintCross(XGabelpos0, YGabelpos0, 10, WindowWithPaintings, 250, 0, 0, LineWidth);
					CHalBase::PaintCross(XGabelpos1, YGabelpos1, 10, WindowWithPaintings, 250, 0, 0, LineWidth);
				}

				if (InspectionActive(InspectionNo + 1) && (Value > MaxValue))
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Gabelmass zu groß = %.3fmm (Max: %.3fmm)", Value, MaxValue);
					snprintf(str_en, sizeof(str_en), "Gap size too big = %.3fmm (Max %.3fmm)", Value, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectGapSizeTooLarge);

					PaintR = 255; PaintG = 0; PaintB = 0;
					CHalBase::PaintCross(XGabelpos0, YGabelpos0, 10, WindowWithPaintings, 250, 0, 0, LineWidth);
					CHalBase::PaintCross(XGabelpos1, YGabelpos1, 10, WindowWithPaintings, 250, 0, 0, LineWidth);
				}

				if (GapResultArrowInside)
					CHalBase::PaintArrow2(XGabelpos1, YGabelpos1, XGabelpos0, YGabelpos0, 10, 10, WindowWithPaintings, PaintR, PaintG, PaintB, LineWidth);
				else
				{
					if (GapDirection == GapDirections::Horizontal)
					{
						CHalBase::PaintArrow(XGabelpos1, min(YGabelpos0, YGabelpos1) - 30, XGabelpos1, min(YGabelpos0, YGabelpos1), 10, 10, WindowWithPaintings, PaintR, PaintG, PaintB, LineWidth);
						CHalBase::PaintArrow(XGabelpos0, max(YGabelpos0, YGabelpos1) + 30, XGabelpos0, max(YGabelpos0, YGabelpos1), 10, 10, WindowWithPaintings, PaintR, PaintG, PaintB, LineWidth);
					}
					else
					{
						CHalBase::PaintArrow(min(XGabelpos0, XGabelpos1) - 30, YGabelpos1, min(XGabelpos0, XGabelpos1), YGabelpos1, 10, 10, WindowWithPaintings, PaintR, PaintG, PaintB, LineWidth);
						CHalBase::PaintArrow(max(XGabelpos0, XGabelpos1) + 30, YGabelpos0, max(XGabelpos0, XGabelpos1), YGabelpos0, 10, 10, WindowWithPaintings, PaintR, PaintG, PaintB, LineWidth);
					}
				}
				sprintf(string1, "%.3fmm", Value);
				CHalBase::PaintText(string1, GapCenterX + GapResultPosX, YGabelpos0 + FontSize + GapResultPosY, WindowWithPaintings, PaintR, PaintG, PaintB, FontSizeGapSize);

				// set data chart
				MeasurementChartSet(MeasurementChartNo, Value, MinValue, MaxValue, MinValue - 0.1, MaxValue + 0.1);

			}


			TimeMeasurementStop();
		}
#pragma endregion

		////////////////////////
		// 4.1 Innenraum hinten
#pragma region STDC_007-3_Particle in inner space of pressure plate
		InspectionNo = 23;
		RejectStatisticGroupNo = 4;
		RejectStatisticEntryNo = 1;
		MeasurementChartNo = 13;
		Paintings = false;
		if (!RejectPole() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();

			// Messwert berechnen
			double r0, c0, phi0, w0, h0 = 0;
			double r1, c1, phi1, w1, h1 = 0;
			r0 = hxa[0].SmallestRectangle2Xld(&c0, &phi0, &w0, &h0);
			r1 = hxa[1].SmallestRectangle2Xld(&c1, &phi1, &w1, &h1);
			HRegion hrInnerArea0;
			HRegion hrInnerArea1;
			HRegion hrInnerArea = HRegion::GenRectangle2(GapCenterY, GapCenterX, GapAngle, Pixelmass_X(KamNr, TargetGapHeight), GapWidth);

			if (GapDirection == GapDirections::Horizontal)
			{
				if (r0 < r1)
				{
					hrInnerArea0 = HRegion::GenRectangle2(r0 - MmToPixel(0.1), GapCenterX, phi0, GapHeight * 2, MmToPixel(0.115));
					hrInnerArea1 = HRegion::GenRectangle2(r1 + MmToPixel(0.1), GapCenterX, phi1, GapHeight * 2, MmToPixel(0.115));
				}
				else
				{
					hrInnerArea0 = HRegion::GenRectangle2(r0 + MmToPixel(0.1), GapCenterX, phi0, GapHeight * 2, MmToPixel(0.115));
					hrInnerArea1 = HRegion::GenRectangle2(r1 - MmToPixel(0.1), GapCenterX, phi1, GapHeight * 2, MmToPixel(0.115));
				}
			}
			else
			{
				if (c0 < c1)
				{
					hrInnerArea0 = HRegion::GenRectangle2(GapCenterY, c0 - MmToPixel(0.1), phi0, GapHeight * 2, MmToPixel(0.115));
					hrInnerArea1 = HRegion::GenRectangle2(GapCenterY, c1 + MmToPixel(0.1), phi1, GapHeight * 2, MmToPixel(0.115));
				}
				else
				{
					hrInnerArea0 = HRegion::GenRectangle2(GapCenterY, c0 + MmToPixel(0.1), phi0, GapHeight * 2, MmToPixel(0.115));
					hrInnerArea1 = HRegion::GenRectangle2(GapCenterY, c1 - MmToPixel(0.1), phi1, GapHeight * 2, MmToPixel(0.115));
				}
			}

			HXLDContArray hxaUnion = UnionXld(hxa);
			HTuple row, column, radius;
			row = hxaUnion.SmallestCircleXld(&column, &radius);
			HRegion hrMiniDom = HRegion::GenCircle(row, column, radius);  // Region for checking inside area if the edges are short.

			MinValue = TargetGapHeight * 0.90;
			double height = PixelToMm(radius[0].D() * 2);

			hrInnerArea = hrInnerArea.Difference(hrInnerArea0);
			hrInnerArea = hrInnerArea.Difference(hrInnerArea1);

			if (height > MinValue)
				hrInnerArea = hrInnerArea.Difference(hrInnerArea.Difference(hrMiniDom.ErosionCircle(ErosionCircleInnerArea)));
			else
				hrInnerArea = hrInnerArea.Difference(hrInnerArea.Difference(hrDomArea.ErosionCircle(ErosionCircleInnerArea)));

			CHalBase::PaintRegion(hrInnerArea, WindowWithPaintings, 0, 0, 255);
			//CHalBase::PaintRegion(hrInnerArea0, WindowWithPaintings, 0, 0, 255);
			//CHalBase::PaintRegion(hrInnerArea1, WindowWithPaintings, 0, 0, 255);
			//CHalBase::PaintRegion(hrMiniDom, WindowWithPaintings, 0, 0, 255);

			//Glättung durch Mittelwertbildung
			HImage himgMean = himgDomFB.ReduceDomain(hrInnerArea).MeanImage(MaskWidthInnerArea, MaskHeightInnerArea);	//Breite und Höhe der Filtermaske
			HRegion hrParticles = himgDomFB.ReduceDomain(hrInnerArea).DynThreshold(himgMean, DynThreshInnerArea, "dark");
			hrParticles = hrParticles.OpeningCircle(2.5);

			HRegionArray hraParticles = hrParticles.Connection();
			//CHalBase::PaintRegionFilled(hraParticles, Anz, 0,0,0);
			hraParticles = hraParticles.SelectShape("width", "and", 12.0, 99999.9);
			//CHalBase::PaintRegionFilled(hraParticles, Anz, 100,100,100);

			Value = 0;

			if (hraParticles.Num() > 0)
			{
				hrParticles = hraParticles.Union1();
				CHalBase::PaintRegionFilled(hrParticles, WindowWithPaintings, 255, 100, 100);

				MaxValue = Beo_MaxErrorAreaInnerArea; // mm²
				MaxValue = BeoSysGetValue(CritNr_STDC_007_PressurePlateInnerArea, SettNr_STDC_007_PressurePlateInnerArea, MaxValue);

				// Messwert berechnen
				double row, column = 0;
				Value = PixelToMm2(hrParticles.AreaCenter(&row, &column));

				if (Value > MaxValue)
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Partikel im Innenraum = %.3fmm² (Max: %.3fmm²)", Value, MaxValue);
					snprintf(str_en, sizeof(str_en), "Particle in inner area = %.3fmm² (Max: %.3fmm²)", Value, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectInnerArea);
				}
			}

			// set data chart
			if (Value > 0)
			MeasurementChartSet(MeasurementChartNo, Value, Hidden, MaxValue, 0.0, MaxValue + 0.1);

			TimeMeasurementStop();
		}
#pragma endregion


		////////////////////////
		// 4.1 Innenraum vorne
#pragma region STDC_007-3_Particle in inner space of pressure plate
		InspectionNo = 24;
		RejectStatisticGroupNo = 4;
		RejectStatisticEntryNo = 1;
		MeasurementChartNo = 13;
		Paintings = false;
		if (!RejectPole() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();

			// Messwert berechnen
			double r0, c0, phi0, w0, h0 = 0;
			double r1, c1, phi1, w1, h1 = 0;
			r0 = hxa[0].SmallestRectangle2Xld(&c0, &phi0, &w0, &h0);
			r1 = hxa[1].SmallestRectangle2Xld(&c1, &phi1, &w1, &h1);
			HRegion hrInnerArea0;
			HRegion hrInnerArea1;
			HRegion hrInnerArea = HRegion::GenRectangle2(GapCenterY, GapCenterX, GapAngle, MmToPixel(TargetGapHeight), GapWidth);

			if (GapDirection == GapDirections::Horizontal)
			{
				if (r0 < r1)
				{
					hrInnerArea0 = HRegion::GenRectangle2(r0 - MmToPixel(0.1), GapCenterX, phi0, GapHeight * 2, MmToPixel(0.115));
					hrInnerArea1 = HRegion::GenRectangle2(r1 + MmToPixel(0.1), GapCenterX, phi1, GapHeight * 2, MmToPixel(0.115));
				}
				else
				{
					hrInnerArea0 = HRegion::GenRectangle2(r0 + MmToPixel(0.1), GapCenterX, phi0, GapHeight * 2, MmToPixel(0.115));
					hrInnerArea1 = HRegion::GenRectangle2(r1 - MmToPixel(0.1), GapCenterX, phi1, GapHeight * 2, MmToPixel(0.115));
				}
			}
			else
			{
				if (c0 < c1)
				{
					hrInnerArea0 = HRegion::GenRectangle2(GapCenterY, c0 - MmToPixel(0.1), phi0, GapHeight * 2, MmToPixel(0.115));
					hrInnerArea1 = HRegion::GenRectangle2(GapCenterY, c1 + MmToPixel(0.1), phi1, GapHeight * 2, MmToPixel(0.115));
				}
				else
				{
					hrInnerArea0 = HRegion::GenRectangle2(GapCenterY, c0 + MmToPixel(0.1), phi0, GapHeight * 2, MmToPixel(0.115));
					hrInnerArea1 = HRegion::GenRectangle2(GapCenterY, c1 - MmToPixel(0.1), phi1, GapHeight * 2, MmToPixel(0.115));
				}
			}


			HXLDContArray hxaUnion = UnionXld(hxa);
			HTuple row, column, radius;
			row = hxaUnion.SmallestCircleXld(&column, &radius);
			HRegion hrMiniDom = HRegion::GenCircle(row, column, radius);

			MinValue = TargetGapHeight * 0.90;
			double height = PixelToMm(radius[0].D() * 2);

			hrInnerArea = hrInnerArea.Difference(hrInnerArea0);
			hrInnerArea = hrInnerArea.Difference(hrInnerArea1);

			if (height > MinValue)
				hrInnerArea = hrInnerArea.Difference(hrInnerArea.Difference(hrMiniDom.ErosionCircle(ErosionCircleInnerArea)));
			else
				hrInnerArea = hrInnerArea.Difference(hrInnerArea.Difference(hrDomArea.ErosionCircle(ErosionCircleInnerArea)));

			CHalBase::PaintRegion(hrInnerArea, WindowWithPaintings, 0, 0, 255);
			//CHalBase::PaintRegion(hrInnerArea0, WindowWithPaintings, 0, 0, 255);
			//CHalBase::PaintRegion(hrInnerArea1, WindowWithPaintings, 0, 0, 255);
			//CHalBase::PaintRegion(hrMiniDom, WindowWithPaintings, 0, 0, 255);

			HImage himgSobel = himgDomFB.ReduceDomain(hrInnerArea).SobelAmp("sum_abs", 3);

			HRegion hrSearch = himgSobel.Threshold(50, 255);
			hrSearch = hrSearch.ShapeTrans("convex");
			CHalBase::PaintRegion(hrSearch, WindowWithPaintings, 0, 150, 0);

			HRegion hrParticles = himgDomFB.ReduceDomain(hrSearch).BinThreshold();

			Value = 0;

			if (hrParticles.Area() > 0)
			{
				CHalBase::PaintRegionFilled(hrParticles, WindowWithPaintings, 0, 250, 0);

				MaxValue = Beo_MaxErrorAreaInnerArea; // mm²
				MaxValue = BeoSysGetValue(CritNr_STDC_007_PressurePlateInnerArea, SettNr_STDC_007_PressurePlateInnerArea, MaxValue);

				// Messwert berechnen
				Value = PixelToMm2(hrParticles.AreaCenter(__nullptr, __nullptr));

				if (Value > MaxValue)
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Partikel im Innenraum = %.3fmm² (Max: %.3fmm²)", Value, MaxValue);
					snprintf(str_en, sizeof(str_en), "Particle in inner area = %.3fmm² (Max: %.3fmm²)", Value, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectInnerArea);

					CHalBase::PaintRegionFilled(hrParticles, WindowWithPaintings, 250, 0, 0);
					CHalBase::PaintRegion(hrSearch, WindowWithPaintings, 150, 0, 0);
				}
			}

			// set data chart
			if (Value > 0)
				MeasurementChartSet(MeasurementChartNo, Value, Hidden, MaxValue, 0.0, MaxValue + 0.1);

			TimeMeasurementStop();
		}
#pragma endregion


		////////////////////////
		// 4.3 GabelFarbe
#pragma region STDC_007-4_Pressure plate coating
		InspectionNo			= 25;
		RejectStatisticGroupNo	= 4;
		RejectStatisticEntryNo	= 2;
		MeasurementChartNo		= 14;
		Paintings				= false;
		if (!RejectPole() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();

			HImage himgR, himgG, himgB, himgOri;
			HImage himgH, himgS, himgV;

			HRegion hrColorCheck = hrDomArea.ErosionCircle(20);

			if (InspectionActive(32))
			{
				HRegion hrInnerArea = HRegion::GenRectangle2(GapCenterY, GapCenterX, GapAngle, MmToPixel(TargetGapHeight), GapWidth);
				hrColorCheck = hrColorCheck.Difference(hrInnerArea);
				//CHalBase::PaintRegion(hrColorCheck, Anz, 0, 0, 255);
			}

			HImage himgGabColor;
			if (InspectionActive(42) || ColorCheckWithoutBrightSpreading) // Suche nur im Spalt
				himgGabColor = himgDom.ReduceDomain(hrColorCheck);
			else
				himgGabColor = himgDomFB.ReduceDomain(hrColorCheck);
			//CHalBase::PaintRegion(hrColorCheck, Anz, 100, 100, 255);

			if (InspectionActive(32)) // Suche nur im Spalt
				himgGabColor = himgDom.ReduceDomain(hrColorCheck);

			// Konvertierung von RGB nach HSV.
			himgR = himgGabColor.AccessChannel(1);
			himgG = himgGabColor.AccessChannel(2);
			himgB = himgGabColor.AccessChannel(3);
			himgH, himgS, himgV;
			himgH = himgR.TransFromRgb(himgG, himgB, &himgS, &himgV, "hsv");
			//himgH.WriteImage("png", 0, "C:\\imgH.png");
			//himgS.WriteImage("png", 0, "C:\\imgS.png");
			//himgV.WriteImage("png", 0, "C:\\imgV.png");

			//himgR.WriteImage("png", 0, "C:\\imgR.png");
			//himgG.WriteImage("png", 0, "C:\\imgG.png");
			//himgB.WriteImage("png", 0, "C:\\imgB.png");

			// Sättigungsskanal ins Zusatzfenster1 kopieren.
			//CHalBase::HImage1ToRgbh(himgS, Zus1, KamNr);
			HRegion hrSaet = himgS.Threshold(MinSatGap, 255);
			hrSaet = hrSaet.OpeningCircle(3.5);
			//CHalBase::PaintRegionFilled(hrSaet, Anz, 100, 100, 255);

			double col = 35 * 0.7; // Umrechnung Wikel auf [0-255]
			HRegion hrColor = himgH.ReduceDomain(hrSaet).Threshold(max(0, col - 20), col + 50);
			//CHalBase::PaintRegionFilled(hrColor, Anz, 255, 100, 100);

			// Nur auswerten wenn die Fläche groß genug ist
			double row, column;
			double areaAU = hrColor.AreaCenter(&row, &column);
			double AreaDomLoch = hrColorCheck.AreaCenter(&row, &column);
			Value = (areaAU / AreaDomLoch) * 100;

			MinValue = MinMaxValueColorCheck; // in %
			MaxValue = MinMaxValueColorCheck; // in %

			if (Value > MaxValue && Artikel.Typ[AP_GapCoating] == GapCoating_SN)
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Falsche Gabelfarbe (AU erkannt) = %.0f%% (Max: %.0f%%)", Value, MaxValue);
				snprintf(str_en, sizeof(str_en), "Wrong gap color (AU found) = %.0f%% (Max: %.0f%%)", Value, MaxValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectGapCoating);

				snprintf(str_de, sizeof(str_de), "AU Gabel erkannt");
				snprintf(str_en, sizeof(str_en), "AU metal detected");
				SelectLanguage(string1, str_de, str_en);
				CHalBase::PaintRegionFilled(hrSaet, WindowWithPaintings, 255, 100, 100);
				CHalBase::PaintRegionFilled(hrColor, WindowWithPaintings, 155, 50, 50);
				CHalBase::PaintText(string1, GapCenterX, GapCenterY - 20, WindowWithPaintings, 255, 0, 0, FontSize);
			}
			if (Value < MinValue && Artikel.Typ[AP_GapCoating] >= GapCoating_AU)
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Falsche Gabelfarbe (SN erkannt) = %.0f%% (Min: %.0f%%)", Value, MaxValue);
				snprintf(str_en, sizeof(str_en), "Wrong gap color (SN found) = %.0f%% (Min: %.0f%%)", Value, MaxValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectGapCoating);

				snprintf(str_de, sizeof(str_de), "SN Gabel erkannt");
				snprintf(str_en, sizeof(str_en), "SN metal detected");
				SelectLanguage(string1, str_de, str_en);
				CHalBase::PaintRegionFilled(hrSaet, WindowWithPaintings, 255, 100, 100);
				CHalBase::PaintRegionFilled(hrColor, WindowWithPaintings, 155, 50, 50);
				CHalBase::PaintText(string1, GapCenterX, GapCenterY - 20, WindowWithPaintings, 255, 0, 0, FontSize);
			}

			if (Artikel.Typ[AP_GapCoating] == GapCoating_SN)
				// set data chart
				MeasurementChartSet(MeasurementChartNo, Value, Hidden, MaxValue);
			else
				// set data chart
				MeasurementChartSet(MeasurementChartNo, Value, MinValue, Hidden);

			TimeMeasurementStop();
		}
#pragma endregion

		if(RejectPoleResult(PoleNoInImage) == PlcRejectGapSizeTooSmall || RejectPoleResult(PoleNoInImage) == PlcRejectGapSizeTooLarge)
			UsingShutterSpecialSetting = false;

		if (!InspectionActive(40))
			UsingShutterSpecialSetting = false;

		if (ShutterSpecialSettingUsed && InspectionActive(39) && PoleNoInImage == 1)	//bei mehreren Polen im Bild nur ein Bild speichern
		{
			snprintf(str_de, sizeof(str_de), "Shutter erhöht %i", Vid[KamNr].Wert[5]);
			snprintf(str_en, sizeof(str_en), "Shutter increased %i", Vid[KamNr].Wert[5]);
			SelectLanguage(string1, str_de, str_en);

			SaveSpecialImage(WindowCamera, "Special image", string1, false, 4);
		}


		// Call additional non-standard inspections
		//ImageEvaluationProcedure_Additionals1()


		//---------------------------  End of pole check  ---------------------------------

#pragma region DO NOT EDIT IN HERE

		// call general features
		if (EndPoleCheck())
			return; // in case of reject and !DontStopAfterFirstReject

	}// end pole loop

	// stop if reject
	if (BeginBlockCheck2())
		return;

#pragma endregion

	//----------------------------  Begin of block check 2  --------------------------------





	//-----------------------------  End of block check 2  ---------------------------------

#pragma region DO NOT EDIT IN HERE

	// call general features
	EndBlockCheck2();
	
} // ImageEvaluationProcedure()

#pragma endregion
