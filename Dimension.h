/*******************************************************************************
**  Project		: BM0xxxxx			                                          **
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
using namespace Dimension;

//-------------------------------	Functions	--------------------------------
void EnableOrDisableInspectionInCamera(int No,bool b)
{
   PFeld[KamNr][No].aktiv = b;
}


/// <summary>
/// Procedure of image evaluation.
/// </summary>
void ImageEvaluationProcedure()
{
	//---------------------------  Begin of block check 1  ----------------------------------

	char roiName[50];

	////////////////////////
	// 1.1 Article is missing
#pragma region
	InspectionNo = 2;
	RejectStatisticGroupNo = 1;
	RejectStatisticEntryNo = 1;
	MeasurementChartNo = 0;
	Paintings = PaintingsBrightFields();
	if (!Reject() && InspectionActive(InspectionNo))
	{
		TimeMeasurementStart();

		// set tolerance
		MinValue = 25; // minimum deviation

		// set ROI
		X1 = ImageWidth() / 4;
		Y1 = ImageHeight() / 4;
		X2 = ImageWidth() * 3 / 4;
		Y2 = ImageHeight() * 3 / 4;

		// generate region and image
		Roi = HRegion::GenRectangle1(Y1, X1, Y2, X2);
		HImage hImg = poHImg->getImg()->ReduceDomain(Roi).Rgb1ToGray();
		
		if (Paintings)
		{
			CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 0, 0, LineWidth * 2); 	// shadow
			CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth); 	// automatically red painted in case of reject
		}

		double deviation = 0.0;
		Roi.Intensity(hImg, &deviation);

		Value = deviation;	// result

		R = 255;
		G = 0;
		B = 255;
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

		// set overlay data (dimension and information)
		OverlayData_SetRectangle(X1, Y1, X2, Y2, "Deviation", Value, 160, 160, 0);

		// set data chart
		MeasurementChartSet(MeasurementChartNo, Value, MinValue);

		TimeMeasurementStop();
	}


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
		// 2.1 Spring is missing
#pragma region
		InspectionNo = 6;
		RejectStatisticGroupNo = 2;
		RejectStatisticEntryNo = 1;
		MeasurementChartNo = 2;
		Paintings = PaintingsEditable1();
#ifdef _Halcon_ShapeModel
		InitSMResult(&oResArticle); // since ShapeModelLib V2.0.21
		if (!Reject() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();
			// set tolerance
			MinValue = 5; // minimum deviation

			// set ROI
			CenterRow = ImageWidth() / 2;
			CenterCol = ImageHeight() / 2;

			switch (Round_dti(Artikel.Nr))
			{
			case 1588905:
				MinValue = 18;
				outerRadius = ImageWidth() * 35.5 / 80;
				innerRadius = ImageWidth() * 33 / 80;
				break;
			case 1588908:
				MinValue = 14.0;
				outerRadius = ImageWidth() * 35.5 / 80;
				innerRadius = ImageWidth() * 33 / 80;
				break;
			case 1588911:
				MinValue = 38;
				outerRadius = ImageWidth() * 21.5 / 80;
				innerRadius = ImageWidth() * 17.5 / 80;
				break;
			case 1588912:
			default:
				MinValue = 44;
				outerRadius = 1500/*ImageWidth() * 22 / 80*/;
				innerRadius = 1000/*ImageWidth() * 18 / 80*/;
				break;
			}

			outerCircle = HRegion::GenCircle(CenterRow, CenterCol, outerRadius);
			innerCircle = HRegion::GenCircle(CenterRow, CenterCol, innerRadius);

			// generate region and image
			Roi = outerCircle.Difference(innerCircle);
			HImage hImg = poHImg->getImg()->ReduceDomain(Roi).Rgb1ToGray();
			if (Paintings)
			{
				CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 0, 0, LineWidth * 2); 	// shadow
				CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth); 	// automatically red painted in case of reject
			}

			double deviation = 0.0;
			Roi.Intensity(hImg, &deviation);

			Value = deviation;	// result

			R = 0;
			G = 255;
			B = 255;
			// check result
			if (Value < MinValue)
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Spring fehlt, Streuung = %.2f (Min %.2f)", Value, MinValue);
				snprintf(str_en, sizeof(str_en), "Spring is missing, deviation = %.2f (Min %.2f)", Value, MinValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringIsMissing, Roi);
			}

			// set overlay data (dimension and information)
			OverlayData_SetRectangle(X1, Y1, X2, Y2, "Deviation", Value, 160, 160, 0);

			// set data chart
			MeasurementChartSet(MeasurementChartNo, Value, MinValue);

			
		}
#pragma endregion

	
		int x = 1200;
		int y = 1800;
		int textSize = 50;
		////////////////////////
		// 2.3 Spring dimension
#pragma region STDC_000-0_TemplateStandardInspection
		InspectionNo = 8;
		RejectStatisticGroupNo = 2;
		RejectStatisticEntryNo = 3;
		MeasurementChartNo = 4;
		Paintings = PaintingsEditable2();
		if (!RejectPole() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();
			HImage hiPImg = *poHImg->getImg();		
			HTuple rowY, colX, imagecX, imagecY;
			int Error;
			float score = 0.4;
			if (artikelNr == 1588905 || artikelNr == 1588908)//400A
			{
				Paintings = PaintingsEditable3();
				Error = Points.FoundPointsOnCircle(Paintings, hiPImg, score, rowY, colX, imagecX, imagecY);
			}
			else //150 250 
			{
				
				Paintings = PaintingsEditable2();	
				Error = Points.FoundPointsOnCircle(Paintings, hiPImg, score, rowY, colX, imagecX, imagecY);
			}
			
			if (Error > 0)
			{
				if (Error == 1)
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "DL process error!");
					snprintf(str_en, sizeof(str_en), "DL process error!");
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringFindError);
				}
				else if (Error == 2)
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Image center found error!");
					snprintf(str_en, sizeof(str_en), "Image center found error!");
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringFindError);
				}
				else if (Error == 3)
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "DL find no points!");
					snprintf(str_en, sizeof(str_en), "DL find no points!");
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringFindError);
				}
				else if (Error == 4)
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Image center points null!");
					snprintf(str_en, sizeof(str_en), "Image center points null!");
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo -1, RejectMessage, PlcRejectSpringDeformation);
				}

			}
			//else if (Error < 0)
			//{
			//	// set reject message
			//	snprintf(str_de, sizeof(str_de), "FindPoints function abnormal!");
			//	snprintf(str_en, sizeof(str_en), "FindPoints function abnormal!");
			//	SelectLanguage(RejectMessage, str_de, str_en);

			//	// set reject
			//	RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringFindError);
			//}
			else if (Error == 0)
			{
				
				if (rowY.Num() > 3)
				{
					HXLDCont hxldCross = HXLDCont::GenContourPolygonXld(rowY, colX);
					double dX, dY, dR, dStartPhi, dEndPhi;
					char szPointOrder[16];
					dY = hxldCross.FitCircleContourXld("geohuber", -1, 0, 0, 3, 1,
						&dX, &dR, &dStartPhi, &dEndPhi, szPointOrder);
					HRegion fitCircle = HRegion::GenCircle(dY, dX, dR);
					actualR = dR; actualX = dX; actualY = dY;
					if (Paintings == 4) CHalBase::PaintRegion(fitCircle, WindowWithPaintings, 0, 255, 255, 5);//7
					HTuple distanceError;
					HTuple mean;
					HTuple max;
					HTuple devation;
					for (int i = 0; i < rowY.Num(); i++)
					{
						double pointToRadius = DistancePP(rowY[i], colX[i], dY, dX);
						distanceError.Append(abs(pointToRadius - dR));								
					}
					
					Value = PixelToMm(dR * 2);
					double MissingValue = 5;
					switch (Round_dti(Artikel.Nr))
					{
					case 1588912:
						MinValue = 7.54;//7.59
						MaxValue = 7.69;
						MissingValue = 8;
						//Value = Value * 0.94271 + 0.495258;
						Value = Value * 0.867478111 + 1.07590708;//new calibration
						break;
					case 1588911:
						MinValue = 7.54;//7.59
						MaxValue = 7.69;
						MissingValue = 8;
						//Value = Value * 0.8563 + 1.1103;
						Value = Value * 0.63130444 + 2.84227507+0.005;//new calibration
						break;
					case 1588908:
						MinValue = 13.66;//13.66
						MaxValue = 13.8;
						MissingValue = 14.1;
						//Value = Value * 0.9767 + 0.39238;
						Value = Value * 0.23866 + 10.4531;////new calibration
						break;
					case 1588905:
					case 1339054:
						MinValue = 7.5 - 0.07;
						MaxValue = 7.5 + 0.07;
						MissingValue = 14.1;
						Value -= 0.015;
						break;
					default://150 250
						MinValue = 7.5 - 0.07;
						MaxValue = 7.5 + 0.07;
						MissingValue = 14.1;
						Value = Value;
						break;
					}

					springDimension = Value;
					double minX = Points.fixedCenterX - 200;
					double maxX = Points.fixedCenterX + 200;
					double minY = Points.fixedCenterY - 200;
					double maxY = Points.fixedCenterY + 200;
					R = 0; G = 255; B = 0;
					if (dX < minX || dX >  maxX || dY< minY || dY >  maxY)
					{
						R = 255; G = 0; B = 0;
						if (dX < minX || dX >  maxX)
						{
							// set reject message
							snprintf(str_de, sizeof(str_de), "Spring center point X: %.1f (MinX %.1f, MaxX %.1f)", dX, minX, maxX);
							snprintf(str_en, sizeof(str_en), "Spring center point X: %.1f (MinX %.1f, MaxX %.1f)", dX, minX, maxX);
						}
						else if (dY< minY || dY >  maxY)
						{
							// set reject message
							snprintf(str_de, sizeof(str_de), "Spring center point Y: %.1f (MinY %.1f, MaxY %.1f)", dY, minY, maxY);
							snprintf(str_en, sizeof(str_en), "Spring center point Y: %.1f (MinY %.1f, MaxY %.1f)", dY, minY, maxY);
						}
						SelectLanguage(RejectMessage, str_de, str_en);

						// set reject
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringDimensionBelowLimit/*, Roi*/);
					}

					sprintf(string1, "centerX: %.3f ", dX);
					CHalBase::PaintText(string1, x, y += textSize, WindowWithPaintings, R, G, B, textSize);
					sprintf(string1, "centerY: %.3f ", dY);
					CHalBase::PaintText(string1, x, y += textSize, WindowWithPaintings, R, G, B, textSize);


					R = 0; G = 255; B = 0;
					// check result
					if (Value < MinValue)
					{
						R = 0; G = 255; B = 0;
						// set reject message
						snprintf(str_de, sizeof(str_de), "Spring dimension %.3f (Min %.3f, Max %.3f)", Value, MinValue, MaxValue);
						snprintf(str_en, sizeof(str_en), "Spring dimension %.3f (Min %.3f, Max %.3f)", Value, MinValue, MaxValue);
						SelectLanguage(RejectMessage, str_de, str_en);

						// set reject
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringDimensionBelowLimit/*, Roi*/);
					}

					if (Value > MaxValue)
					{
						if (Value > MissingValue)
						{
							R = 0; G = 255; B = 0;
							// set reject message
							snprintf(str_de, sizeof(str_de), "Spring missing");
							snprintf(str_en, sizeof(str_en), "Spring missing");
							SelectLanguage(RejectMessage, str_de, str_en);

							// set reject
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringIsMissing/*, Roi*/);
						}
						else
						{
							R = 0; G = 255; B = 0;
							// set reject message
							snprintf(str_de, sizeof(str_de), "Spring dimension %.3f (Min %.3f, Max %.3f)", Value, MinValue, MaxValue);
							snprintf(str_en, sizeof(str_en), "Spring dimension %.3f (Min %.3f, Max %.3f)", Value, MinValue, MaxValue);
							SelectLanguage(RejectMessage, str_de, str_en);

							// set reject
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringDimensionAboveLimit/*, Roi*/);
						}
					}
					if (Value > MissingValue)
					{
						R = 0; G = 255; B = 0;
						// set reject message
						snprintf(str_de, sizeof(str_de), "Spring missing");
						snprintf(str_en, sizeof(str_en), "Spring missing");
						SelectLanguage(RejectMessage, str_de, str_en);

						// set reject
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringIsMissing/*, Roi*/);
					}

					snprintf(string1, sizeof(string1), "%.3f", Value);
					CHalBase::PaintText(
						string1,			// text
						ImageWidth() / 2,
						ImageHeight() / 2,	    // position (x, y)
						WindowWithPaintings,// WindowWithPaintings
						R, G, B,			// color (R/G/B)
						100,					// size
						HW_FONT_ARIAL,		// font
						true,				// bold
						false,				// kusiv
						Round_dti(0),		// angle
						eAlignCenter,		// horizontal transition
						eAlignBottom,		// vertical transition
						true);

					double tempArr[3];
					tempArr[0] = Value; tempArr[1] = MinValue; tempArr[2] = MaxValue;
					DataSave.SaveArrayTxt(
						tempArr,				// Passed array of floating point values
						3,						// Number of values to pass from array
						false,					// True - convert dot separated value to coma separated value,
						",",					// Separation mark between values
						"C1.txt",			    // Name of final txt file
						"a",
						true,					// True - split file with current date catalog (YY_MM_DD) active,
						true,					// True - start text line with time stamp HH:MM:SS
						"d:\\Statistics");			// Directory where to store file

					R = 0; G = 255; B = 0;
					int greaterThanMean = Points.TupleProcess(distanceError, mean, max, devation);					
					//todo:
					//400A 
					if (artikelNr == 1588908 || artikelNr == 1588905)  MaxValue = 15;
					if (artikelNr == 1588911 || artikelNr == 1588912)  MaxValue = 25;
					Value = devation[0].D();
					MaxValue =20;
					if (devation[0].D() > MaxValue)
					{
						R = 255; G = 0; B = 0;
						snprintf(str_de, sizeof(str_de), "Spring devation %.3f ( Max %.3f)", Value, MaxValue);
						snprintf(str_en, sizeof(str_en), "Spring devation %.3f ( Max %.3f)", Value, MaxValue);
						SelectLanguage(RejectMessage, str_de, str_en);

						// set reject
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringDimensionBelowLimit/*, Roi*/);
					}
					
					sprintf(string1, "maxError: %.3f ", max[0].D());
					CHalBase::PaintText(string1, x, y += textSize, WindowWithPaintings, R, G, B, textSize);
					sprintf(string1, "meanError: %.3f ", mean[0].D());
					CHalBase::PaintText(string1, x, y += textSize, WindowWithPaintings, R, G, B, textSize);
					sprintf(string1, "devationError: %.3f ", devation[0].D());
					CHalBase::PaintText(string1, x, y += textSize, WindowWithPaintings, R, G, B, textSize);
					sprintf(string1, "greaterThanMeanNum: %.i ", greaterThanMean);
					CHalBase::PaintText(string1, x, y += textSize, WindowWithPaintings, R, G, B, textSize);
				}
				else
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Points No. on the circle is less than 4!");
					snprintf(str_en, sizeof(str_en), "Points No. on the circle is less than 4!");
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringDimensionBelowLimit/*, Roi*/);
				}
			}
			

#pragma region outside STDI_001_ColorDistinction

			TimeMeasurementStop();

		}
#pragma endregion

		////////////////////////
		// 2.2 Spring deformation
#pragma region
		InspectionNo = 7;
		RejectStatisticGroupNo = 2;
		RejectStatisticEntryNo = 2;
		MeasurementChartNo = 3;
		Paintings = PaintingsBrightFields();
		if (!RejectPole() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();		
			outerCircle = HRegion::GenCircle(actualY, actualX, actualR- 60 );
			innerCircle = HRegion::GenCircle(actualY, actualX, actualR - 300);
			Roi = outerCircle.Difference(innerCircle);
			HImage hImg = poHImg->getImg()->ReduceDomain(Roi).Rgb1ToGray();
			if (Paintings)
			{
				CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 0, 0, LineWidth * 2); 	// shadow
				CHalBase::PaintRegion(Roi, WindowWithPaintings, 255, 0, 255, LineWidth); 	// automatically red painted in case of reject
			}
			Hlong UsedThreshold1;
			HRegion binRegion = hImg.BinaryThreshold("max_separability", "dark", &UsedThreshold1);
			binRegion = SortRegionArea(binRegion.Connection(), true)[0];
			CenterRow = binRegion.SmallestRectangle2(&CenterCol, &AngleRad,&Length1, &Length2);
			HRegion r =HRegion::GenRectangle2(CenterRow, CenterCol, AngleRad, Length1, Length2);
			CHalBase::PaintRegion(r, WindowWithPaintings, 0, 255, 0);

			double vLongRow = -sin(AngleRad);
			double vLongCol = cos(AngleRad);

			// 短边1的中点
			double shortRow1 = CenterRow + Length1 * vLongRow;
			double shortCol1 = CenterCol + Length1 * vLongCol;

			// 短边2的中点
			double shortRow2 = CenterRow - Length1 * vLongRow;
			double shortCol2 = CenterCol - Length1 * vLongCol;

			CHalBase::PaintCross(shortCol1, shortRow1,10, WindowWithPaintings, 255, 0, 0, 10);
			CHalBase::PaintCross(shortCol2, shortRow2,10, WindowWithPaintings, 255, 0, 0, 10);
			double dis1 = DistancePP(actualX, actualY, shortCol1, shortRow1);
			double dis2 = DistancePP(actualX, actualY, shortCol2, shortRow2);
			Value = (dis1 < dis2) ? dis1 : dis2;

			
			

			
			MaxValue = actualR -300;
			R = 255;
			G = 255;
			B = 0;
			// check result
			if (Value < MaxValue)
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Spring deformation, Streuung = %.2f (Max %.2f)", Value, MaxValue);
				snprintf(str_en, sizeof(str_en), "Spring deformation, deviation = %.2f (Max %.2f)", Value, MaxValue);
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringDeformation, Roi);
			}

			// set overlay data (dimension and information)
			OverlayData_SetRectangle(X1, Y1, X2, Y2, "Deviation", Value, 160, 160, 0);

			// set data chart
			MeasurementChartSet(MeasurementChartNo, Value, MinValue);

			TimeMeasurementStop();
		}
#endif
#pragma endregion

		////////////////////////
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
