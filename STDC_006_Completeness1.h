/*******************************************************************************
**  Project		: BM039612                                                    **
********************************************************************************
** Filename		: Completeness.h
** Programmer	: E. Bogatz
** Date			: 08.08.2022
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
using namespace nSTDC_006_Completeness;

//-------------------------------	Functions	--------------------------------

/// <summary>
/// Simplified deviation check via shape model roi with info parameters: [un-/expected];[min];[max]
/// </summary>
/// <param name="roiName">Name of the roi to be checked.</param>
/// <param name="smResult">The shape model result which contains '<paramref name="roiName"/>'.</param>
/// <param name="plcResult">The error result on reject.</param>
/// <param name="paramMinMinValue">The minimum accepted value for minimum deviation [min].</param>
/// <param name="paramMinMaxValue">The maximum accepted value for minimum deviation [min].</param>
/// <param name="paramMaxMinValue">The minimum accepted value for maximum deviation [max].</param>
/// <param name="paramMaxMaxValue">The maximum accepted value for maximum deviation [max].</param>
/// <param name="expected"><c>true</c> if <paramref name="roiName"/> should be found (article parameter depended). Gets overwritten by the shape model info parameter setup [un-/expected]. Error if article parameter and info parameter setup differ. <c>nullptr</c> to ignore</param>
/// <returns><c>true</c> if the check was successful, false on reject.</returns>
//bool checkViaDeviation(SMLibResult smResult, char roiName[50], const BYTE plcResult, double paramMinMinValue, double paramMinMaxValue, double paramMaxMinValue, double paramMaxMaxValue, bool* expected);
bool checkViaDeviation(HImage HImg,								// HImage auf dem gearbeitet wird
	char smName[50],											// Name of the model (xxx)
	SMLibResult smResult,										// Result of the shape model search (for position, angle, ...)
	char roiName[50],											// Name of region to check deviation ("MoldCheckx")
	const BYTE plcResult,										// in case of error -> number for return to PLC
	double paramMinDev_MinValue, double paramMinDev_MaxValue,	// MIN Deviation: Erlaubter Bereich (Min, Max)
	double paramMaxDev_MinValue, double paramMaxDev_MaxValue,	// MAX Deviation: Erlaubter Bereich (Min, Max) für MAX Deviation
	bool* expected,												// set to 'nullptr'
	int ModelIndex,												// Model Index für das Ziel Modell (bei max 8: 0 - 7)
	int LocalMeasurementChartNo,								// WerBereich Nummer für das Ergebnis der Streuung (only MC 1-6 reserved)
	int LocicalUnderfilledNumber);								// Loigsche Nummer der Unterspritzung an diesem Artikel (fortlaufend von 1 - 6 max.)


bool MeasurePoint(HImage image, HRegion region, double angle, double sigma, double threshold, char transition[20], char select[20], int paintings,
	double* x, double* y)
{
	try
	{
		*x = *y = 0;
		
		// Abmasse berechnen
		double row, col, phi, length1, length2;
		row = region.SmallestRectangle2(&col, &phi, &length1, &length2);

		double col2, row2;
		PointOnVector(col, row, angle, length1, col2, row2);
		if (paintings > 1)
			CHalBase::PaintArrow(col, row, col2, row2, 25, 25, WindowWithPaintings, 0, 0, 255, 2);
		    
		// HMeasure-Objekt erzeugen
		HMeasure hMeasure(row, col, angle, length1, 10, image.Width(), image.Height(), "nearest_neighbor");
		//double angle = MathHelper.RadToDeg(phi);
		/*HRegion reg = HRegion::GenRectangle2(row,col, phi, length1, length2);
		CHalBase::PaintRegion(reg, WindowWithPaintings, 0, 255, 255, 25);
		return true;*/
		// Kanten ermitteln
		HTuple rowEdge, colEdge, amplitude, distance;
		rowEdge = hMeasure.MeasurePos(image,
			sigma,		//Sigma der Gaußglättung.
			threshold,	//Minimale Amplitude einer Kante.
			transition,	//Transition: all, negative, positive
			"all",		//Auswahl der Kantenpunkte: all, first, last
			&colEdge, &amplitude, &distance);
	
		// paint all found points
		for (int i = 0; i < rowEdge.Num(); i++)
			if (paintings > 1)
				CHalBase::PaintCross(Round_dti(colEdge[i].D()), Round_dti(rowEdge[i].D()), 10, WindowWithPaintings, 155, 0, 155, 2);

		// paint selected point
		if (rowEdge.Num() > 0)
		{
			if (strcmp(select, "last") == 0)
			{
				if (paintings)
					CHalBase::PaintCross(Round_dti(colEdge[rowEdge.Num() - 1].D()), Round_dti(rowEdge[rowEdge.Num() - 1].D()), 15, WindowWithPaintings, 255, 0, 255, 3);
				*x = colEdge[rowEdge.Num() - 1].D();
				*y = rowEdge[rowEdge.Num() - 1].D();
			}
			else
			{
				if (paintings)
					CHalBase::PaintCross(Round_dti(colEdge[0].D()), Round_dti(rowEdge[0].D()), 15, WindowWithPaintings, 255, 0, 255, 3);
				*x = colEdge[0].D();
				*y = rowEdge[0].D();
			}
		}
		else
		{
			// error
			return true;
		}
	}
	catch (HException& except)
	{
		// error
		return true;
	}
	// ToDo: Sortierung erster letzer Messpunkt
	//HTuple index = colEdge. .SortIndex();

	// Rückgabekoordinaten des gefundenen Messpunkts
	/*y = rowEdge[0].D;
	x = colEdge[0].D;*/

	return false; // no error
}

// reset statistic for automatic dummy check, replaced by ApplyArticleChange()
void resetStatisticDummyCheck()
{
	while (Kontrolldatei_Eintrag_aktiv)
		Sleep(10);
	Kontrolldatei_Eintrag("Dummyprüfung", 0);

	// Artikelparameterauswahl: Änderungen übernehmen
	Auswahl_Gehaeusefarbe(1, Artikel.Typ[1]);

	// Ausschusszähler löschen
	Ausschusszaehler_loeschen(1, 1);	// Artikelwechsel (Automatisch)

	Einstelldatei_schreiben();
}


// manage automatic dummy check
void automaticDummyCheck(int dummyNo, 			// >0 Change articledata for dummy, 0 = switch back to saved article date
						 int posX, int posY,	// Text-Position DummyNo inisde image
						 int fontSize)			// Text size DummyNo inside image
{
	static bool init = true;
	char string1[200];

	if (init)
	{
		// delete article buffers during first measurement
		ZeroMemory(&articleBeforeDummyCheck, sizeof(articleBeforeDummyCheck));
		articleNoBeforeDummyCheck = 0;							// Article number before dummy check was started
		strcpy(articleTextBeforeDummyCheck, "");				// Article text before dummy check was started
		init = false;
	}

	if (dummyNo > 0)
	{
		//snprintf(str_de, sizeof(str_de), "Dummy %i", dummyNo);
		//snprintf(str_en, sizeof(str_en), "Dummy %i", dummyNo);
		//SelectLanguage(string1, str_de, str_en);
		//CHalBase::PaintText(string1, posX, posY, WindowWithPaintings, 250, 250, 250, fontSize, HW_FONT_ARIAL, true, false, 0, eAlignLeft, eAlignTop, true);

		// dummy article configuration
		if (!dummyCheckActive)
		{
			// 1. turn on dummy check.
			// Used to turn it off before measurement of next normal article
			dummyCheckActive = true;

			// 2. buffer article data
			for (size_t i = 0; i <= MaxEinzel; i++)
				articleBeforeDummyCheck[i] = Artikel.Typ[i];
			articleNoBeforeDummyCheck = Artikel.Nr;					// Article number before dummy check was started
			strcpy(articleTextBeforeDummyCheck, Artikel.Text);		// Article text before dummy check was started

			// 3. set dummy article data
			if ((dummyNo >= 10) && (dummyNo <= 14))					// BM039361 - all dummies: PT 2,5 QUATTRO YEGN
			{
				Artikel.Typ[1] = ArticleColor_GNYE;
				Artikel.Typ[2] = AP2_QUATTRO;
			}
			// 4. set article
			article.Set(ArticleNo(), ArticleParameter(1), ArticleParameter(2), int(PusherTypes::XTShape), poSMs, ShapeModelDirectory);

			// 5. apply the Article change
			snprintf(string1, sizeof(string1), "C%i - Start dummy check", Kameraliste[CameraNo].Nr);
			ApplyArticleChange(string1);	// Text with Trigger-Source
			//resetStatisticDummyCheck();
		}
	}

	// reset article data to data before dummy check
	else if (dummyCheckActive)
	{
		dummyCheckActive = false;
		for (size_t i = 0; i <= MaxEinzel; i++)
			Artikel.Typ[i] = articleBeforeDummyCheck[i];
		article.Set(ArticleNo(), ArticleParameter(1), ArticleParameter(2), int(PusherTypes::XTShape), poSMs, ShapeModelDirectory);
		Artikel.Nr = articleNoBeforeDummyCheck;					// Article number before dummy check was started
		strcpy(Artikel.Text, articleTextBeforeDummyCheck);		// Article text before dummy check was started

		// Apply the Article change
		snprintf(string1, sizeof(string1), "C%i - End of dummy check", Kameraliste[CameraNo].Nr);
		ApplyArticleChange(string1);	// Text with Trigger-Source
		//resetStatisticDummyCheck();
	}
}




// Remember the results of the pushers
#define MaxResults 5
#define MaxResultData 10
typedef struct
{
	char Text[20];		// Name of Result data
	int OK;				// Result is OK
	int XPos;			// XPosition for result text
	int YPos;			// YPosition for result text
} sResultData;

typedef struct								// Ausgabe-Bild
{
	int Amont;								// Number of data
	sResultData Data[MaxResultData+1];		// Data for result & text
} sResult;
sResult PartResult[MaxResults+1];			// Results of single parts e.g. pusher with a lot of checks


int InitResultPusher()
{
	int ResultPart = 0;	// ==> Pusher
	for (int i = 0; i <= MaxResultData; i++)
	{
		PartResult[ResultPart].Amont = 0;
		snprintf(PartResult[ResultPart].Data[i].Text, sizeof(PartResult[ResultPart].Data[i].Text), "");
		PartResult[ResultPart].Data[i].OK = 0;
		PartResult[ResultPart].Data[i].XPos = -1;
		PartResult[ResultPart].Data[i].YPos = -1;
	}
	return 1;
}


int SetResultPusher(int PusherNo, int OK, int XPos=(-1), int YPos=(-1));

int SetResultPusher(int PusherNo, int OK, int XPos/*=(-1)*/, int YPos/*=-(1)*/)
{
	int ResultPart = 0;

	if ((PusherNo >= 0) && (PusherNo <= MaxResultData))
	{
		PartResult[ResultPart].Data[PusherNo].OK = OK;
		if ((XPos >= 0) && (YPos >= 0))
		{
			snprintf(PartResult[ResultPart].Data[PusherNo].Text, sizeof(PartResult[ResultPart].Data[PusherNo].Text), "Pusher %i", PusherNo);
			PartResult[ResultPart].Amont++;
			PartResult[ResultPart].Data[PusherNo].XPos = XPos;
			PartResult[ResultPart].Data[PusherNo].YPos = YPos;
		}
		return 1;
	}
	else
	{
		return 0;
	}
}


int ShowResultPusher(int iWindow, int iAlignX, int iAlignY, bool iShowOnlyRejects)
{
	int ResultPart = 0;	// ==> Pusher
	for (int i = 0; i <= MaxResultData; i++)
	{
		if (i <= PartResult[ResultPart].Amont)
		{
			int R = 0;
			int G = 255;
			int B = 0;
			if (!PartResult[ResultPart].Data[i].OK)
			{
				R = 255; 
				G = 0;
				B = 0;
			}

			if (!iShowOnlyRejects || !PartResult[ResultPart].Data[i].OK)
			{
				//CHalBase::PaintRegion(pusher, WindowWithPaintings, R, G, B, 1);
				snprintf(str_de, sizeof(str_de), "%s", PartResult[ResultPart].Data[i].Text);
				snprintf(str_en, sizeof(str_en), "%s", PartResult[ResultPart].Data[i].Text);
				SelectLanguage(string1, str_de, str_en);

				if ((PartResult[ResultPart].Data[i].XPos >= 0) &&
					(PartResult[ResultPart].Data[i].YPos >= 0))
				{
					CHalBase::PaintText(string1, PartResult[ResultPart].Data[i].XPos, PartResult[ResultPart].Data[i].YPos,
						iWindow, R, G, B, FontSize, HW_FONT_ARIAL, true, false, 0, iAlignX, iAlignY, true);//eAlignRight, eAlignCenter, true);
				}
			}
		}
	}
	return 1;
}


/// <summary>
/// Procedure of image evaluation.
/// </summary>
void ImageEvaluationProcedure()
{
	// Definitions for Information text (FontSize an Positions)
	int InformationFontSize		= Wertkontrolle_Int((int)(0.40 * FontSize), 11, 20);	// min. FontSize = 11, max. 20
	// Jeder benutzt andere Einzeichnungs Positionen. Warum nicht eine Startsposition von außen vorgeben?
	int Information_RowSize		= Round_dti(InformationFontSize * 1.10);				// Row Y-distance
	int Information_YPosiion	= ImageHeight() - 20 - InformationFontSize;				// Text Y-start position: from Image Bottom to Top row
	//int Information_XPosiion = TargetPositionX;										// Text X-start position: Center of expected pole
	int Information_XPosiion	= ImageWidth() / 2;										// Text X-start position: Image center

	automaticDummyCheck(dummyNo, MmToPixel(11.00), MmToPixel(1.00), FontSize);			// Drawing DummyNo internally deactivated

	
	//---------------------------  Begin of block check 1  ----------------------------------


	// common vars
	char roiName[50];
	char roiNameNumbered[50];
	int MeasurementChartNo2 = 0, MeasurementChartNo3 = 0, MeasurementChartNo4 = 0;

	// Display DummyNo inside camera image
	if (dummyNo > 0)
	{
		snprintf(str_de, sizeof(str_de), "DummyNr. %i", dummyNo);
		snprintf(str_en, sizeof(str_en), "DummyNo %i", dummyNo);
		SelectLanguage(string1, str_de, str_en);
		CHalBase::PaintText(string1, 25, 40, WindowWithPaintings, 250, 250, 250, (int)(FontSize * 1.50), HW_FONT_ARIAL, true, false, 0, eAlignLeft, eAlignTop, true);
		CHalBase::PaintText(string1, Information_XPosiion, Information_YPosiion, WindowWithPaintings, 120, 120, 120, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignCenter, eAlignBottom, true);
		Information_YPosiion -= Information_RowSize;
	}


#pragma region BlockCheck

	////////////////////////
	// article setup

	// Show warning information in image
	double textPosY = FontSize;
	if (article.Warning())
	{
		stringstream warningList(article.WarningText());
		size_t warningNo = 1;
		while (warningList.good())
		{
			string warning;
			getline(warningList, warning, '\n');
			snprintf(string1, sizeof(string1), "[%i] %s", warningNo, warning.c_str());
			CHalBase::PaintText(string1, 50, textPosY, WindowWithPaintings, 200, 200, 0, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignLeft, eAlignTop, true);
			textPosY += FontSize;
			warningNo++;
		}
	}

	// cancel process on invalid article data
	if (article.Error())
	{
		string error = article.ErrorText();
		CompletenessArticle::ReplaceAll(error, "\n"," # ");
		snprintf(RejectMessage, sizeof(RejectMessage), "%s", error.c_str());
		RejectSet(0, 0, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());

		// Show error information in image
		stringstream errorList(article.ErrorText());
		size_t errorNo = 1;
		while (errorList.good())
		{
			string error;
			getline(errorList, error, '\n');
			snprintf(string1, sizeof(string1), "[%i] %s", errorNo, error.c_str());
			CHalBase::PaintText(string1, 50, textPosY, WindowWithPaintings, 200, 0, 0, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignLeft, eAlignTop, true);
			textPosY += FontSize;
			errorNo++;
		}
		return;
	}
	automaticDummyCheck(dummyNo, MmToPixel(11.00), MmToPixel(1.00), FontSize);


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

		int PointNo = 1;	// Variable for actual MeasurementPoint
		AngleDeg = 0;
		bool PECalibration_active = false;
		double PECalibration_YOffset	= 0.824;	// mm
		PECalibration_YOffset			= 0.844;	// mm	calibration 2023-12-05
		PECalibration_YOffset			= 0.822;	// mm	calibration 2023-12-08
		if (InspectionActive(35)) PECalibration_active = true;

		////////////////////////
		// 1.1 Search housing
#pragma region
		InspectionNo			= 2;
		RejectStatisticGroupNo	= 1;
		RejectStatisticEntryNo	= 1;
		MeasurementChartNo		= 0;
		Paintings				= PaintingsEditable1();
#ifdef _Halcon_ShapeModel
		SMLibResult oResHousing;
		InitSMResult(&oResHousing); // since ShapeModelLib V2.0.21
		if (!RejectPole())
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("1.1 search housing started", CameraNo);
			if (ShapeModelUsed)
			{
				// set minimum/maximum score
				MinValue = 70;
				if (!InspectionActive(InspectionNo)) // disable score check
					MinValue = 0;

				if(ArticleParameter(1)== ArticleColor_BK)  MinValue = 60;
				HImage  shapeImage;
				HImage originalImage = *poHImg->getImg();
				HImage shapeImageR, shapeImageG, shapeImageB;
				shapeImageR = originalImage.Decompose3(&shapeImageG, &shapeImageB);
				switch (ArticleParameter(1))
				{
				    case ArticleColor_BU:shapeImage = shapeImageB; break;
					case ArticleColor_RD:
					case ArticleColor_BN:
						shapeImage = shapeImageR; break;
					case ArticleColor_GNYE:
					case ArticleColor_WH:
				    case ArticleColor_GY:
					case ArticleColor_VT:
					case ArticleColor_YE:
					case ArticleColor_OG:
					case ArticleColor_BK:
						shapeImage = originalImage.Rgb1ToGray();;
						break;				  
				}


				// set ROI
				X1		= TargetPositionX;
				Y1		= TargetPositionY;
				Width	= MmToPixel(5.00);
				Height	= MmToPixel(6.00);
				DWORD		ShapeModelActivateHousing = 0x01;	// Activated ShapeModel(s) of a group (0 activates all)
				//ShapeModelActivate = 0x01;
				// set name of ShapeModel
				ShapeModelActive = article.ShapeArticleName();
				
			
				// search ShapeModel
				int num = poSMs->find(shapeImage,
					                ShapeModelActive,					// name of ShapeModel
									X1, Y1,						// search position X/Y, default=0/0 for using ShapeModel settings
									Width, Height,				// search position width/height, default=0/0 for using ShapeModel settings
					ShapeModelActivateHousing,			// activate ShapeModel (0 for using ShapeModel settings)
									true);						// true:	ShapeModel should be found (MinValue used)
																// false:	ShapeModel should not be found (MaxValue used)

				// get ShapeModel results
				oResHousing = poSMs->res(ShapeModelActive,		// name of ShapeModel
									MeasurementChartNo,			// measurement chart number for score, default=-1
									MinValue, //MaxValue,		// minimum / maximum score
									Paintings,					// activate paintings, default=1
									LineWidth,					// width paintings, default=1
									0);							// index, default=0
				vector_angle_to_rigid(0, 0, 0, oResHousing.y, oResHousing.x, oResHousing.angle, &HomMat2DImage);
				if (ArticleParameter(3) != AP3_no_PE_Foot)
				{
					hom_mat2d_invert(HomMat2DImage, &HomMat2DInvert);
					hom_mat2d_compose(HomMat2DPattern, HomMat2DInvert, &HomMat2DCompose);
				}
				
				// ShapeModel was not found
				if (poSMs->Err(ShapeModelActive))
				{
					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, poSMs->getErr(ShapeModelActive), PlcRejectHousingIsMissing, poSMs->getSearchRegion(ShapeModelActive));
				}
				else
				{
					// apply found position
					PoleCenterX = Round_dti(oResHousing.x_ref_point);
					PoleCenterY = Round_dti(oResHousing.y_ref_point);
					AngleDeg = oResHousing.angle_deg;

					// Display housing angle
					snprintf(string1, sizeof(string1), "Angle Housing = %.2f°", AngleDeg);
					TextLegendPaintings(PoleCenterX, PoleCenterY, -120, 150, string1, (int)(0.50*FontSize), 200,255,200);
				}

				// Enhanced shape model result paintings (scoreResult)
				if (Paintings > 0)
					EnhancedModelResultPaintings(oResHousing, MmToPixel(+3.00), MmToPixel(-1.50), /*X1, Y1,*/ true, MinValue, WindowWithPaintings, 100, R, G, B);
			}
			else // ShapeModel searching is not activated
			{
				// set reject message
				snprintf(str_de, sizeof(str_de), "Shapemodel Suche erforderlich!");
				snprintf(str_en, sizeof(str_en), "Shapemodel search required!");
				SelectLanguage(RejectMessage, str_de, str_en);

				// set reject
				RejectSet(0, 0, RejectMessage, PlcRejectMalfunction);
			}
			Logger::getInstance().LogInfo("1.1 search housing stop", CameraNo);
			TimeMeasurementStop();
		}
#endif
#pragma endregion



		////////////////////////
		// 1.2 Housing color
#pragma region STDI_001_ColorDistinction
		InspectionNo 			= 3;
		RejectStatisticGroupNo 	= 1;
		RejectStatisticEntryNo 	= 2;
		if (!RejectPole() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("1.2 Housing color started", CameraNo);
			// roi defined?
			RoiName = "ColorDistinctionHousing";
			poSMs->getRoiAndParams(oResHousing, RoiName, Roi);
			if (Roi.IsEmpty())
			{
				snprintf(str_de, sizeof(str_de), "ROI '%s' nicht definiert!", RoiName);		//only ROI name
				snprintf(str_en, sizeof(str_en), "ROI '%s' isn't defined!", RoiName);		//only ROI name
				SelectLanguage(RejectMessage, str_de, str_en);
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, Roi);
			}
			else
			{
				// check color with integrated automatic teach function (Ctrl + X)
				if (CheckColor(1, Roi, ArticleParameter(1)))
				{
					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectWrongColor, Roi);
				}

#pragma region outside STDI_001_ColorDistinction
				// paintings
				int textAlignmentX = eAlignCenter;
				int textAlignmentY = eAlignCenter;

				// use passed info parameters: TextBelow to paint the text below the region, default above the region
				HTuple infoParams = poSMs->getRoiInfo(oResHousing, RoiName);
				Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);
				if (infoParams.Num() > 0 && infoParams.IsString() && infoParams[0] == HTuple("TextBelow"))
				{
					textAlignmentY = eAlignTop;
					Y1 = Row2 + MmToPixel(0.50);
				}
				else
				{
					textAlignmentY = eAlignBottom;
					Y1 = Row1 - MmToPixel(0.50);
				}
				snprintf(str_de, sizeof(str_de), "Gehäusefarbe");
				snprintf(str_en, sizeof(str_en), "Housing color");
				SelectLanguage(string1, str_de, str_en);
				CHalBase::PaintText(string1, Col1 / 2 + Col2 / 2, Y1, WindowWithPaintings, R, G, B, FontSize, HW_FONT_ARIAL, true, false, 0, textAlignmentX, textAlignmentY, true);
#pragma endregion outside STDI_001_ColorDistinction
				Logger::getInstance().LogInfo("1.2 housing clolor stop", CameraNo);
				TimeMeasurementStop();
			}
		}
#pragma endregion STDI_001_ColorDistinction


// # # # # # # # # # # # # # # # 
// # # # # # # # # # # # # # # # 
//		////////////////////////
//		// 1.3 Housing underfilled V2++ - 2. step simple deviation for each region (with shape position correction)
//#pragma region
//		//#ifdef AP1_Underfilled
//		InspectionNo = 4;
//		RejectStatisticGroupNo = 1;
//		RejectStatisticEntryNo = 3;
//		MeasurementChartNo = 27;
//		Paintings = PaintingsEditable3();
//		int FilterWindowNo = WindowCamera;			// Result Filter Window for undermold deviation check for YE /GN
//		if (!RejectPole() && InspectionActive(InspectionNo))
//		{
//			TimeMeasurementStart();
//
//			// Image to use different channels (BU / RD special)
//			HImage HImageUnderfilledCheck;
//			HImageUnderfilledCheck = *poHImg->getImg();
//
//			if (ArticleParameter(AP_ArticleColor) == ArticleColor_BU)
//			{
//				HImageUnderfilledCheck = poHImg->getImg()->AccessChannel(3);
//				snprintf(string1, sizeof(string1), "Check underfillings only in BLUE channel");
//				CHalBase::PaintText(string1, Information_XPosiion, Information_YPosiion, WindowWithPaintings, 120, 120, 120, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignCenter, eAlignBottom, true);
//				Information_YPosiion -= Information_RowSize;
//			}
//			if (ArticleParameter(AP_ArticleColor) == ArticleColor_RD)
//			{
//				HImageUnderfilledCheck = poHImg->getImg()->AccessChannel(1);
//				snprintf(string1, sizeof(string1), "Check underfillings only in RED channel");
//				CHalBase::PaintText(string1, Information_XPosiion, Information_YPosiion, WindowWithPaintings, 1200, 120, 120, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignCenter, eAlignBottom, true);
//				Information_YPosiion -= Information_RowSize;
//			}
//
//			if ((ArticleParameter(AP_ArticleColor) == ArticleColor_GNYE) && InspectionActive(37))
//			{
//				for (size_t filterNo = 0; filterNo < 10; filterNo++)
//				{
//					snprintf(roiNameNumbered, sizeof(roiNameNumbered), "ColorFilter%i", filterNo + 1);
//					poSMs->getRoiAndParams(oResHousing, roiNameNumbered, Roi);
//					Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);
//
//					if (Roi.IsEmpty())
//					{
//						snprintf(str_de, sizeof(str_de), "ROI für ColorFilter fehlt!");
//						snprintf(str_en, sizeof(str_en), "ROI of ColorFilter is missing!");
//						SelectLanguage(RejectMessage, str_de, str_en);
//						//RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
//						//break;
//					}
//					else
//					{
//
//						// apply special color to gray filter for yellow / green housing
//						Filter_HSL_stretching(1, CameraNo,	// Number of filter (only for display), Camera number
//							Col1 - 10, Row1 - 10,			// Top left corner (Measurement area)
//							Col2 + 10, Row2 + 10,			// Bottom right (Measurement area)
//							0,								// Min Hue angle [0-360] (lowest border for HUE-stretching)
//							360,							// Max Hue angle [0-360] (highest border for HUE-stretching)
//							0,								// Min Saturation [0-100] (lowest border for SAT-stretching)
//							15,								// Max Saturation [0-100] (highest border for SAT-stretching)
//							5,								// Min Brightness [0-100] (lowest border for LUM-stretching)
//							// 2024-03-20: get a little smoother brightness for YE/GN housings.
//							65,//55,								// Max Brightness [0-100] (highest border for LUM-stretching)
//							0,								// SAT Result invert. (0 = High Sat -> white Pixels, 1 = Low Sat -> white Pixels)
//							1,								// 0=Add HSL-Results, 1=Multiply HSL-Results
//							WindowCamera,					// Number of source image
//							WindowAdditional3,				// Number of target image
//							WindowAdditional3,				// Number of display window for paintings (normally the same as target image)
//							1);								// Paintings
//						FilterWindowNo = WindowAdditional3;	// set filter to result target window
//					}
//				}
//
//				HImageUnderfilledCheck = CHalBase::RgbhToHImage(FilterWindowNo, CameraNo, 0);
//				//poHImg->setImg(hiPImgPins);
//				//poHImg->setImg(4, FilterWindowNo);
//
//				//HImageUnderfilledCheck = poHImg->getImg()->AccessChannel(1);
//				//snprintf(string1, sizeof(string1), "Check underfillings only in RED channel");
//				//CHalBase::PaintText(string1, Information_XPosiion, Information_YPosiion, WindowWithPaintings, 1200, 120, 120, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignCenter, eAlignBottom, true);
//				//Information_YPosiion -= Information_RowSize;
//			}
//
//			size_t underfilledNo = 0;	// Number of underfilled zone (starts every time at 1)
//			bool ResultOK = 0;			// for check deviation
//			int UNumber = 0;			// Target undermold check model index (e.g. at max 8: 0 - 7)
//			int LocalMeasurementChartNo = MeasurementChartNo - 1;	// for shape score result
//
//			snprintf(roiName, sizeof(roiName), "UnderfilledPosition");
//			do
//			{
//				// set color for following paintings
//				G = 255; R = B = 0;
//				snprintf(roiNameNumbered, sizeof(roiName), "%s%i", roiName, ++underfilledNo);
//				if (poSMs->findROI(oResHousing, roiNameNumbered) >= 0)
//				{
//					// Seperate Shape search (for exact position of underfilled check area)
//					SMLibResult oResUndermold;
//					InitSMResult(&oResUndermold); // since ShapeModelLib V2.0.21
//
//					LocalMeasurementChartNo++;
//
//					if (ShapeModelUsed)
//					{
//						Hlong r1, c1, r2, c2;
//						// set minimum/maximum score
//						MinValue = 70;
//						//if (!InspectionActive(InspectionNo)) // disable score check
//							//MinValue = 0;
//
//						poSMs->getRoiAndParams(oResHousing, roiNameNumbered, Roi);
//						r1 = Roi.SmallestRectangle1(&c1, &r2, &c2);
//						CenterRow = Roi.SmallestRectangle2(&CenterCol, nullptr, &Length1, &Length2);
//
//						// set ROI
//						X1 = (int)CenterCol;
//						Y1 = (int)CenterRow;
//						Width = MmToPixel(2.00);
//						Height = MmToPixel(2.00);
//
//						// set name of ShapeModel
//						//ShapeModelActive = SM_Housing;
//						//ShapeModelActive = article.ShapeArticleName();
//						ShapeModelActive = SM_Undermold;
//
//						// use passed info parameters: TextBelow to paint the text below the region, default above the region
//						HTuple infoParams = poSMs->getRoiInfo(oResHousing, roiNameNumbered);
//						DumpTuple(infoParams);
//						//Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);
//						
//						if (infoParams.Num() > 0 && infoParams.IsString())
//						{
//							UNumber = atoi(infoParams[0]);
//							if (UNumber ==  1) ShapeModelActivate = 0x001;
//							if (UNumber ==  2) ShapeModelActivate = 0x002;
//							if (UNumber ==  3) ShapeModelActivate = 0x004;
//							if (UNumber ==  4) ShapeModelActivate = 0x008;
//							if (UNumber ==  5) ShapeModelActivate = 0x010;
//							if (UNumber ==  6) ShapeModelActivate = 0x020;
//							if (UNumber ==  7) ShapeModelActivate = 0x040;
//							if (UNumber ==  8) ShapeModelActivate = 0x080;
//							if (UNumber ==  9) ShapeModelActivate = 0x100;
//							if (UNumber == 10) ShapeModelActivate = 0x200;
//							if (UNumber == 11) ShapeModelActivate = 0x400;
//							if (UNumber == 12) ShapeModelActivate = 0x800;
//						}
//
//						// search ShapeModel
//						int numMatches = poSMs->find(
//							HImageUnderfilledCheck,		// search image
//							ShapeModelActive,			// name of ShapeModel
//							X1, Y1,						// search position X/Y, default=0/0 for using ShapeModel settings
//							Width, Height,				// search position width/height, default=0/0 for using ShapeModel settings
//							ShapeModelActivate,			// activate ShapeModel (0 for using ShapeModel settings)
//							true);						// true:	ShapeModel should be found (MinValue used)
//														// false:	ShapeModel should not be found (MaxValue used)
//
//						// get ShapeModel results
//						oResUndermold = poSMs->res(ShapeModelActive,		// name of ShapeModel
//							LocalMeasurementChartNo,	// measurement chart number for score, default=-1
//							MinValue, //MaxValue,		// minimum / maximum score
//							Paintings,					// activate paintings, default=1
//							LineWidth,					// width paintings, default=1
//							0);							// index, default=0
//
//						// in case of no found shape model: set shape seach position for enhanced model paintings
//						//oResUndermold.x_ref_point = oResUndermold.x = X1;
//						//oResUndermold.y_ref_point = oResUndermold.y = Y1;
//
//						// ShapeModel was found
//						if (!poSMs->Err(ShapeModelActive))
//						{
//							// check position befor additional dev check
//							//PoleCenterX = Round_dti(oResUndermold.x_ref_point);
//							//PoleCenterY = Round_dti(oResUndermold.y_ref_point);
//							//AngleDeg = oResUndermold.angle_deg;
//							char roiNameCheck[50];
//							snprintf(roiNameCheck, sizeof(roiNameCheck), "MoldCheck%i", UNumber);
//
//							char smName[100];
//							snprintf(smName, sizeof(smName), "%s", ShapeModelActive);
//
//							int MAX_MCNumber = 6;
//							int MCNumber = 0 + underfilledNo;		// for deviation result
//							if (MCNumber > MAX_MCNumber) MCNumber = MAX_MCNumber;
//
//							// Paintings inside function checkViaDeviation()
//							ResultOK = checkViaDeviation(HImageUnderfilledCheck,
//								smName,											// Name of the model (xxx)
//								oResUndermold,									// Result of the shape model search (for position, angle, ...)
//								roiNameCheck,									// Name of region to check deviation ("MoldCheckx")
//								PlcRejectHousingUnderfillError,					// in case of error -> number for return to PLC
//								0, 20,											// MIN Deviation: Erlaubter Bereich (Min, Max)
//								0, 100,											// MAX Deviation: Erlaubter Bereich (Min, Max) für MAX Deviation
//								nullptr,										// set to 'nullptr'
//								UNumber - 1,									// Target undermold check model index (e.g. at max 8: 0 - 7)
//								MCNumber,										// Measurement char for result of deviation (only MC 1-6 reserved)
//								underfilledNo);									// locigal number for undermolding check at this articel (ongoing from 1 - 6 max.)
//						}
//						else // ShapeModel was not found
//						{
//							// set reject
//							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, poSMs->getErr(ShapeModelActive), PlcRejectHousingUnderfillError, poSMs->getSearchRegion(ShapeModelActive));
//						}
//
//						// Enhanced shape model result paintings (scoreResult)
//						if (Paintings > 0)
//						{
//							if (!poSMs->Err(ShapeModelActive))
//							{
//								HRegion roiSearchTemp = HRegion::GenEmptyRegion();
//								roiSearchTemp = poSMs->getSearchRegion(ShapeModelActive);		// Modelname (Index)
//								Row1 = roiSearchTemp.SmallestRectangle1(&Col1, &Row2, &Col2);
//							}
//
//							if (oResUndermold.x_ref_point == 0)
//							{
//								//oResUndermold.x_ref_point = oResUndermold.x = X1;
//								//oResUndermold.y_ref_point = oResUndermold.y = Y1;
//							}
//							EnhancedModelResultPaintings(oResUndermold, MmToPixel(-3.00), MmToPixel(+1.50), X1, Y1, true, MinValue, WindowWithPaintings, 100, R, G, B);
//						}
//					}
//					else // ShapeModel searching is not activated
//					{
//						// set reject message
//						snprintf(str_de, sizeof(str_de), "Shapemodel Suche erforderlich!");
//						snprintf(str_en, sizeof(str_en), "Shapemodel search required!");
//						SelectLanguage(RejectMessage, str_de, str_en);
//
//						// set reject
//						RejectSet(0, 0, RejectMessage, PlcRejectMalfunction);
//					}
//					MeasurementChartNo++;
//				}
//				else
//				{
//					if (underfilledNo <= 1)
//					{
//						// Error: no region for UnderFilledPosition defined
//						snprintf(string1, sizeof(string1), "Shape Model '%s': No region '%s' defined", article.ShapeArticleName(), roiNameNumbered);
//						// set reject
//						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, string1, PlcRejectHousingIsMissing, PoleCenterX - 10, PoleCenterY - 10, PoleCenterX + 10, PoleCenterY + 10);
//						//snprintf(roiNameNumbered, sizeof(roiName), "%s%i", roiName, ++underfilledNo);
//					}
//
//					break;
//				}
//			} while (1);//!RejectPole());
//
//			ShapeModelActivate = 0;
//
//			TimeMeasurementStop();
//		}
////#endif AP_Underfilled
//#pragma endregion


		////////////////////////
		// 2.1 Current bar missing
#pragma region
		InspectionNo			= 7;
		RejectStatisticGroupNo	= 2;
		RejectStatisticEntryNo	= 1;
		MeasurementChartNo		= 8;
		MeasurementChartNo2		= MeasurementChartNo + 1;	// Type
		Paintings				= PaintingsEditable1();
		if (!RejectPole() && (InspectionActive(InspectionNo) || InspectionActive(InspectionNo + 1)))
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("2.1 currentbar started", CameraNo);
			SMLibResult oResCurrentBar;
			InitSMResult(&oResCurrentBar);
			snprintf(roiName, sizeof(roiName), "%s", article.RoiCurrentBar());
			bool checked = false;
			size_t currentBarNo = 0;
			HRegion roiCurrentBar = HRegion::GenEmptyRegion();
			do
			{
				snprintf(roiNameNumbered, sizeof(roiNameNumbered), "%s%i", roiName, ++currentBarNo);
				poSMs->getRoiAndParams(oResHousing, roiNameNumbered, roiCurrentBar);
				if (!roiCurrentBar.IsEmpty())
				{
					checked = true;

					// set minimum/maximum score
					MinValue = 70;
					if (!InspectionActive(InspectionNo)) // disable score check
					{
						MinValue = 0;
					}

					// set name of ShapeModel
					ShapeModelActive = article.ShapeCurrentBarName(currentBarNo - 1);

					// search ShapeModel
					poSMs->find(ShapeModelActive,					// name of ShapeModel
										roiCurrentBar,				// search region
										ShapeModelActivate,			// activate ShapeModel (0 for using ShapeModel settings)
										true);						// true:	ShapeModel should be found (MinValue used)
																	// false:	ShapeModel should not be found (MaxValue used)

					// get ShapeModel results
					oResCurrentBar = poSMs->res(ShapeModelActive,	// name of ShapeModel
										MeasurementChartNo,			// measurement chart number for score, default=-1
										MinValue, //MaxValue,		// minimum / maximum score
										Paintings,					// activate paintings, default=1
										LineWidth,					// width paintings, default=1
										0);							// index, default=0

					// ShapeModel was not found
					if (poSMs->Err(ShapeModelActive))
					{
						// set reject
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, poSMs->getErr(ShapeModelActive), PlcRejectCurrentBarError, poSMs->getSearchRegion(ShapeModelActive));
					}
					else
					{
						////////////////////////
						// 2.2 Current bar type
						size_t typeNo = 0;
						snprintf(roiName, sizeof(roiName), "%s", article.RoiCurrentBarType());
						do
						{
							snprintf(roiNameNumbered, sizeof(roiNameNumbered), "%s%i", roiName, ++typeNo);
							poSMs->getRoiAndParams(oResCurrentBar, roiNameNumbered, Roi);
							if (!Roi.IsEmpty())
							{
								HImage metalImg = poHImg->getImg()->ReduceDomain(Roi).Rgb1ToGray();
								double deviation = 0, intensity = 0;
								intensity = Roi.Intensity(metalImg, &deviation);
								HRegion currentbarRegion = metalImg.Threshold(160, 255);
								//CHalBase::PaintRegionFilled(currentbarRegion, WindowWithPaintings, 0, 0, 200);
								Value = PixelToMm2(currentbarRegion.Area());												
								MinValue = 168; MaxValue = 4;
								if (ArticleParameter(2) == AP2_QUATTRO_2P || ArticleParameter(2) == AP2_TWIN_1P || ArticleParameter(2) == AP2_1P|| ArticleParameter(2) == AP2_QUATTRO_2P_PE || ArticleParameter(2) == AP2_TWIN_1P_PE || ArticleParameter(2) == AP2_1P_PE)
								{
									MinValue = 85;//140
									MaxValue = 2.0;
								}
								if (InspectionActive(InspectionNo + 1) && ((intensity < MinValue)|| Value < MaxValue))
								{
									if ((intensity < MinValue))
									{
										snprintf(str_de, sizeof(str_de), "Strombalkentyp: intensity %.2f  (Min %.2f)", intensity, MinValue);
										snprintf(str_en, sizeof(str_en), "Current bar type:intensity %.2f  (Min %.2f)", intensity, MinValue);
										SelectLanguage(RejectMessage, str_de, str_en);
										RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectCurrentBarError, Roi);
										CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth);
									}
									if (Value < MaxValue)
									{
										snprintf(str_de, sizeof(str_de), "Strombalkentyp: area %.2f mm^2 (Min %.2f)", Value, MaxValue);
										snprintf(str_en, sizeof(str_en), "Current bar type:area %.2f mm^2 (Min %.2f)", Value, MaxValue);
										SelectLanguage(RejectMessage, str_de, str_en);
										RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectCurrentBarError, Roi);
										CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth);
									}
								}

								// set data chart
								MeasurementChartSet(MeasurementChartNo2, Value, MinValue, Hidden, 0.0, MinValue + 0.8);

								// paintings
								CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 0, 200, 1);
								CHalBase::PaintRegion(currentbarRegion, WindowWithPaintings, R, G, B, 1);
							}
						} while (!Roi.IsEmpty() && !RejectPole());
/*	// No current bar type detection in BM039361
						////////////////////////
						// 2.2 Current bar type (deviation)
						typeNo = 0;
						snprintf(roiName, sizeof(roiName), "%s", article.RoiCurrentBarType());
						snprintf(roiName, sizeof(roiName), "TypeDeviation");
						do
						{
							snprintf(roiNameNumbered, sizeof(roiNameNumbered), "%s%i", roiName, ++typeNo);
							poSMs->getRoiAndParams(oResCurrentBar, roiNameNumbered, Roi);
							if (!Roi.IsEmpty())
							{
								// generate region and image
								HImage hiPImg = poHImg->getImg()->ReduceDomain(Roi);
								hiPImg = hiPImg.Rgb1ToGray(); // use grey channel

								// check metal length
								double deviation = 0.0;
								double brightness = 0.0;
								brightness = Roi.Intensity(hiPImg, &deviation);
								double Value = deviation; // result

								MinValue =  0.00;
								MaxValue = 10.00;
								if (InspectionActive(InspectionNo + 1) && Value < MinValue)
								{
									snprintf(str_de, sizeof(str_de), "Strombalkentyp: Streuung %.2f (Min:%.2f)", Value, MinValue);
									snprintf(str_en, sizeof(str_en), "Current bar type: deviation %.2f (Min:%.2f)", Value, MinValue);
									SelectLanguage(RejectMessage, str_de, str_en);
									RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectCurrentBarNotFound, Roi);
									CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth);
								}

								if (InspectionActive(InspectionNo + 1) && Value > MaxValue)
								{
									snprintf(str_de, sizeof(str_de), "Strombalkentyp: Streuung %.2f (Max %.2f)", Value, MaxValue);
									snprintf(str_en, sizeof(str_en), "Current bar type: deviation %.2f (Max %.2f)", Value, MaxValue);
									SelectLanguage(RejectMessage, str_de, str_en);
									RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectCurrentBarNotFound, Roi);
									CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth);
								}

								// set data chart
								MeasurementChartSet(MeasurementChartNo2, Value, MinValue, MaxValue, MinValue, MaxValue + 0.8);

								// paintings
								CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, 1);
								//CHalBase::PaintRegion(metal, WindowWithPaintings, R, G, B, 1);
							}
						} while (!Roi.IsEmpty() && !RejectPole());
*/
					}	// End of if (ShapeModelUsed)

					// paintings
					snprintf(str_de, sizeof(str_de), "Strombalken %i", currentBarNo);
					snprintf(str_en, sizeof(str_en), "CurrentBar %i", currentBarNo);
					SelectLanguage(string1, str_de, str_en);
					CenterRow = roiCurrentBar.SmallestRectangle2(&CenterCol, nullptr, &Length1, &Length2);
					CHalBase::PaintText(string1, CenterCol, CenterRow + Length1 + FontSize, WindowWithPaintings, R, G, B, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignCenter, eAlignCenter, true);
					if (Paintings > 0)
					{
						double dX1, dY1; roiCurrentBar.AreaCenter(&dY1, &dX1); X1 = (int)dX1; Y1 = (int)dY1;
						EnhancedModelResultPaintings(oResCurrentBar, Round_dti(CenterCol), Round_dti(CenterRow + Length1 + FontSize * 3.0),/* X1, Y1,*/ false, MinValue, WindowWithPaintings, 100, R, G, B);
					}
				}
			} while (!roiCurrentBar.IsEmpty() && !RejectPole());

			// current bar checked
			if (!checked)
			{
				snprintf(str_de, sizeof(str_de), "ROI für Strombalkenprüfung fehlt!");
				snprintf(str_en, sizeof(str_en), "ROI of current bar check is missing!");
				SelectLanguage(RejectMessage, str_de, str_en);
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
			}
			Logger::getInstance().LogInfo("2.1 currentbar stop", CameraNo);
			TimeMeasurementStop();
		}
#pragma endregion


		// Reset all pusher result text
		InitResultPusher();


		////////////////////////
		// 4.1 Pusher missing (and also pusher color)
#pragma region
		InspectionNo			= 17;
		RejectStatisticGroupNo	= 4;
		RejectStatisticEntryNo	= 1;
		MeasurementChartNo		= 11;
		bool FindPusher = true;
		if (!InspectionActive(17) && !InspectionActive(18) &&!InspectionActive(19) &&!InspectionActive(20) &&!InspectionActive(21))
		{
			FindPusher = false;
		}
		if (!RejectPole() && InspectionActive(InspectionNo))/* FindPusher)*/// 	// always filter pusher and generate region
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("4.1 pusher missing started", CameraNo);
			for (size_t pusherNo =article.FirstPusher(); pusherNo < article.PushersCount() && !RejectPole(); pusherNo++)
			{
				// skip 'not inserted' pushers
				if (article.Pushers(pusherNo)->Type() == PusherTypes::NotInserted)
					continue;

				// get pusher ROI
				snprintf(roiNameNumbered, sizeof(roiNameNumbered), "%s%i", Pusher::RoiName().c_str(), pusherNo + 1);
				poSMs->getRoiAndParams(oResHousing, roiNameNumbered, Roi);
				if (Roi.IsEmpty())
				{
					snprintf(str_de, sizeof(str_de), "ROI für Pusher %i fehlt!", pusherNo + 1);
					snprintf(str_en, sizeof(str_en), "ROI of pusher %i is missing!", pusherNo + 1);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
					break;
				}
				else
				{
					// filter pusher by HSV parameters (pusher region is saved inside article)
					HRegion* pusher = article.Pushers(pusherNo)->FilterPusher2(poHImg->getImg(), Roi);
					HRegion pusher1 = article.Pushers(pusherNo)->Region();

					if (InspectionActive(Debug_MODE))CHalBase::PaintRegionFilled(pusher1, WindowWithPaintings, 0, 255, 0);
			
					if (pusher->IsEmpty())
					{
						if (InspectionActive(InspectionNo))
						{
							snprintf(str_de, sizeof(str_de), "Pusher fehlt / Farbe nicht definiert!");
							snprintf(str_en, sizeof(str_en), "Pusher missing / undefined color!");
							SelectLanguage(RejectMessage, str_de, str_en);
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPusherError, Roi);
							CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth);
						}
					}
					else//*ErosionCircle(MmToPixelDouble(0.10));DilationCircle(MmToPixelDouble(0.10));
					{	
						if (ArticleParameter(1) == ArticleColor_YE)
					    {
							*pusher = SortRegionArea(pusher->Connection(), true)[0].ErosionRectangle1(1,5);
					    }
						else if(ArticleParameter(1) == ArticleColor_OG )
						{
							*pusher = SortRegionArea(pusher->Connection(), true)[0].ErosionRectangle1(5, 5);/*ClosingCircle(MmToPixelDouble(0.10)).OpeningCircle(MmToPixelDouble(0.20))*/;
							
							
						}
					    else
					    {
							//*pusher = SortRegionArea(pusher->Connection(), true)[0].ErosionRectangle1(1, 5).DilationRectangle1(5,1);
							*pusher = SortRegionArea(pusher->Connection(), true)[0].ClosingCircle(MmToPixelDouble(0.20)).OpeningCircle(MmToPixelDouble(0.20));
					    }
						
						Value = PixelToMm2(pusher->Area());
						MinValue = article.Pushers(pusherNo)->MinSize();
						MaxValue = article.Pushers(pusherNo)->MaxSize();
					
						

						// check pusher size
						if (InspectionActive(InspectionNo) && (Value < MinValue || Value > MaxValue))
						{
							snprintf(str_de, sizeof(str_de), "Pusher%ifläche: %.2f mm² (Min %.2f, Max %.2f)", pusherNo + 1,Value, MinValue, MaxValue);
							snprintf(str_en, sizeof(str_en), "Pusher%i area: %.2f mm² (Min %.2f, Max %.2f)", pusherNo + 1 ,Value, MinValue, MaxValue);
							SelectLanguage(RejectMessage, str_de, str_en);
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPusherError, Roi);
							CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth);
							// Area error: Show pusher pixels in additional image
							HImage pusherImg = HImage::GenImageConst("byte", ImageWidth(), ImageHeight());
							HRegion pusher = article.Pushers(pusherNo)->Region();
							if (InspectionActive(Debug_MODE)) pusherImg.OverpaintRegion(pusher, 255, "fill");						// fill in pixels of pusher
							if (InspectionActive(Debug_MODE)) pusherImg = pusherImg.MeanImage(3, 3);
							if (InspectionActive(Debug_MODE)) CHalBase::HImage1ToRgbh(pusherImg, WindowAdditional2, CameraNo, 1);	// paint in red
							if (InspectionActive(Debug_MODE)) CHalBase::PaintRegion(Roi, WindowAdditional2, R, G, B, LineWidth);	// add outlines of pusher search region
							
						}

						// set data chart
						MeasurementChartSet(MeasurementChartNo, Value, MinValue, MaxValue, MinValue - 2.0, MaxValue + 2.0);
					}


					// paintings
					//int iLineWidth = LineWidth - 1; if (iLineWidth < 1) iLineWidth = 1;
					//CHalBase::PaintRegion(pusher, WindowWithPaintings, R, G, B, iLineWidth);
					//snprintf(str_de, sizeof(str_de), "Pusher %i", pusherNo + 1);
					//snprintf(str_en, sizeof(str_en), "Pusher %i", pusherNo + 1);
					//SelectLanguage(string1, str_de, str_en);

					//// Set complete pusher data
					//CenterRow = Roi.SmallestRectangle2(&CenterCol, nullptr, &Length1, &Length2);
					//SetResultPusher(pusherNo + 1, !Reject(), Round_dti(CenterCol - (Length2 * 0.0)), Round_dti(CenterRow - Length1 - (FontSize * 0.50)));

					//{
					//	// DisplayScale limited
					//	double displayScale = 100;
					//	if (displayScale < 10) displayScale = 10;
					//	if (displayScale > 500) displayScale = 500;
					//	double InternalScale = 0.01 * displayScale;


					//	double DisplayValue = Value;
					//	//DisplayValue = 12.00;

					//	Display_ResultValueBar(CameraNo,				// Kamera Nummer
					//		WindowWithPaintings,					// Bild, in dass das Mass eingetragen wird

					//		// Position
					//		CenterCol,					// X-Position Bezugspunkt 
					//		CenterRow,					// Y-Position Bezugspunkt 
					//		-100,						// X-Abstand zum Bezugspunkt (genutzte Position, wenn !VersatzVonRef)
					//		- 50,						// Y-Abstand zum Bezugspunkt (genutzte Position, wenn !VersatzVonRef)
					//		true,						// Falsch wenn die Versatzpositionen als Absolutpositionen ohne den Bezugspunkt verwendet werden sollen

					//		// Design
					//		1,							// Art der Anzeige (0: RT/GT/BT für Text & Balken, 1: Farbe nach Ergebnis berechnen)
					//		(int)(FontSize * 10.5 * InternalScale),						// Breite des Balkens (-1: Standard, >=1: Breite vorgeben)
					//		FontSize,					// Schriftgröße (Zeichensatz)
					//		255, 255, 255,				// Farbe des Textes (R/G/B)

					//		// Values / Borders
					//		DisplayValue,						// ScoreResult vom Model 0-100%
					//		"mm²",						// Text für Einheit (Standard: %)
					//		MinValue,					// Minimum erlaubter Wert (UTG)
					//		MaxValue,					// Minimum erlaubter Wert (UTG)
					//		8.00,//Mittelwert(MinValue, MaxValue),			// Normaler Wert (bei +/- Verteilung ist der Standard 0)
					//		MinValue - 0.50,			// Anzeige MinWert 0-100%
					//		MaxValue + 0.50,			// Anzeige MaxWert 0-100%

					//		// zusätzliche Texte
					//		string1,			// Name der Modell-Suche
					//		"Area");		// Name der Kontrolle
					//}
				}
			}	// End of loop pusher
			Logger::getInstance().LogInfo("4.1 pusher missing stop", CameraNo);
			TimeMeasurementStop();
		}
#pragma endregion



		////////////////////////
		// 3.1 Spring missing/double
#pragma region
		InspectionNo = 11;
		RejectStatisticGroupNo = 3;
		RejectStatisticEntryNo = 1;
		MeasurementChartNo = 10;
		Paintings = PaintingsEditable3();
		if (!RejectPole() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("3.1 Spring missing/double started", CameraNo);
			// create red channel filter image
		   /* poHImgSpring = new CHal_HImage(CameraNo);
		     poHImgSpring->setImg(1);*/

			HImage ImgSpring = *poHImg->getImg();
			HImage ImgSpringR, ImgSpringG, ImgSpringB;
			ImgSpringR = ImgSpring.Decompose3(&ImgSpringG, &ImgSpringB);

			//if (article.HousingColor() == HousingColors::OG)
				//poHImgSpring->setImg(4);
			if (InspectionActive(Debug_MODE)) CHalBase::HImage1ToRgbh(ImgSpringR, WindowAdditional1, CameraNo, 0);
			  // poHImgSpring->paint(WindowAdditional1);
		

			//if (article.HousingColor() == HousingColors::OG)
			//	poHImgSpring->setImg(4);
			//if (InspectionActive(Debug_MODE)) poHImgSpring->paint(WindowAdditional1);

			snprintf(roiName, sizeof(roiName), "%s", article.RoiNameSpring());
			bool checked = false;
			size_t springNo = 0;			
			do
			{
				snprintf(roiNameNumbered, sizeof(roiNameNumbered), "%s%i", roiName, ++springNo);
				poSMs->getRoiAndParams(oResHousing, roiNameNumbered, Roi);				
				if (!Roi.IsEmpty())
				{
					// cut pusher region out of spring ROI if present
					if (article.Pushers(springNo - 1)->Type() != PusherTypes::NotInserted)
					{
						double pusherAreaCenterRow, pusherAreaCenterCol;
						HRegion pusher = article.Pushers(springNo - 1)->Region();
						pusher.AreaCenter(&pusherAreaCenterRow, &pusherAreaCenterCol);
						Roi.AreaCenter(&CenterRow, &CenterCol);
						Value = PixelToMm(DistancePP(CenterCol, CenterRow, pusherAreaCenterCol, pusherAreaCenterRow));
						MaxValue = 5.30;

						// spring too far away from pusher (to detect not fitting pusher numbers)
						if ((Value > MaxValue) && (pusherAreaCenterRow > 0) && (pusherAreaCenterCol > 0))
						{
							snprintf(str_de, sizeof(str_de), "Abstand Feder ROI zum Pusher: %.2f mm (Max %.2f)", Value, MaxValue);
							snprintf(str_en, sizeof(str_en), "Distance spring ROI from pusher: %.2f mm (Max %.2f)", Value, MaxValue);
							SelectLanguage(RejectMessage, str_de, str_en);
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringError, Roi);

							// Distance not OK: show distance with arrow
							CHalBase::PaintArrow2(CenterCol, CenterRow, pusherAreaCenterCol, pusherAreaCenterRow, 10, 10, WindowWithPaintings, R, G, B, LineWidth + 1);
						}
						else
						{
							Roi = Roi.Difference(pusher.DilationCircle(5.0));
							//Roi = Roi.Difference(pusher.DilationCircle(6.0));	// Test only

							// Distance not OK: show distance with arrow
							if (InspectionActive(Debug_MODE))
								CHalBase::PaintArrow2(CenterCol, CenterRow, pusherAreaCenterCol, pusherAreaCenterRow, 10, 10, WindowAdditional1, 0, 255, 0, LineWidth + 1);
						}
					}
					HImage hiPImg = poHImg->getImg()->ReduceDomain(Roi);
					hiPImg = ScaleImageMinMax(hiPImg, 40, 100).Rgb1ToGray();
					HRegion spring = hiPImg.Threshold(180, 255);
					//CHalBase::PaintRegionFilled(spring, WindowWithPaintings, R, G, B);	// spring region
					//CHalBase::PaintRegionFilled(spring, WindowAdditional2, R, G, B);	// spring region
					Value = PixelToMm2(spring.Area());
					
					//// get roi edges in filter image (without pusher)
					checked = true;
					//HImage* imgDir = new HImage();
					//HRegion spring = poHImgSpring->getImg()->ReduceDomain(Roi).EdgesImage(imgDir, "lanser2", 1.0, "nms", 10, 40).Threshold(25, 150);

					//// Help spring edge detection
					//// Threshold: The higher the first value is, the stronger edges have to be in order to be detected as spring edges (default: 0, 255)
					//// For article PTTB 1.5, the lower value had to be increased to 25 because if the spring is missing, a housing contour (or the transition to the outer pusher) can be seen.
					//// The value direct behind 'lanser2': lower values ensure more smoothing in edge detection (default: 1.00)
					//// BM039222 reduced to 0.75

					//if (imgDir != NULL)
					//{
					//	delete imgDir;
					//	imgDir = NULL;
					//}
					//todo:
					//Value = PixelToMm2(spring.Area());
					MinValue = 0.10;//0.09
					//if(ArticleParameter(1) == ArticleColor_GNYE) MinValue = 0.40;
					MaxValue = 0.95;	// to detect doubled springs
					if (ArticleParameter(1) == ArticleColor_OG)
					{
						MaxValue =1.00;
					}
					
					if (Value < MinValue || Value > MaxValue)
					{
						snprintf(str_de, sizeof(str_de), "Feder Konturfläche: %.2f mm² (Min %.2f, Max %.2f)", Value, MinValue, MaxValue);
						snprintf(str_en, sizeof(str_en), "Spring contour area: %.2f mm² (Min %.2f, Max %.2f)", Value, MinValue, MaxValue);
						SelectLanguage(RejectMessage, str_de, str_en);
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectSpringError, Roi);
					}

					// set data chart
					MeasurementChartSet(MeasurementChartNo, Value, MinValue, MaxValue, 0, MaxValue + 0.20);//10

					//3.2 check Spring position
					if (!RejectPole() && InspectionActive(InspectionNo + 1))
					{
						HTuple springInfoParams = poSMs->getRoiInfo(oResHousing, roiNameNumbered);
						const char* depthKeys[] = { "", "minDepth", "maxDepth" };
						int depthIdxList[] = { 0, 1, 2 };
						int depthKeyCnt = sizeof(depthKeys) / sizeof(depthKeys[0]);
						bool depthParamMiss = true;
						bool depthParamIncomplete = false;
						const char* badDepthParam = "Info";

						if (springInfoParams.Num() > 0 && springInfoParams.IsString() && HTuple(springInfoParams[0]).Strlen() > 0)
						{
							HTuple springParts = HTuple(springInfoParams[0]).Split(";");
							depthParamMiss = false;

							for (int k = 0; !depthParamMiss && !depthParamIncomplete && k < depthKeyCnt; k++)
							{
								if (springParts.Num() <= depthIdxList[k])
								{
									depthParamMiss = true;
									badDepthParam = (k == 0) ? "L/R" : depthKeys[k];
									break;
								}

								HTuple seg = HTuple(springParts[depthIdxList[k]]).Split(",");
								bool keyOk = false;
								if (k == 0)
								{
									keyOk = (seg.Num() >= 1 && (seg[0] == HTuple("L") || seg[0] == HTuple("R")));
								}
								else
								{
									string keyWithSpaceText = string(" ") + depthKeys[k];
									HTuple keyWithSpace = HTuple(keyWithSpaceText.c_str());
									keyOk = (seg.Num() >= 1 && (seg[0] == HTuple(depthKeys[k]) || seg[0] == keyWithSpace));
								}

								if (!keyOk)
								{
									depthParamMiss = true;
									badDepthParam = (k == 0) ? "L/R" : depthKeys[k];
									break;
								}

								int valueIndex = (k == 0) ? 0 : 1;
								if (seg.Num() <= valueIndex || HTuple(seg[valueIndex]).Strlen() == 0)
								{
									depthParamIncomplete = true;
									badDepthParam = (k == 0) ? "L/R" : depthKeys[k];
									break;
								}

								if (k == 1) MinValue = seg[1].D();
								if (k == 2) MaxValue = seg[1].D();
							}
						}

						if (depthParamMiss || depthParamIncomplete)
						{
							char str_de[256] = { 0 }, str_en[256] = { 0 };
							if (depthParamIncomplete)
							{
								snprintf(str_de, sizeof(str_de), "ROI '%s': Info-Parameter '%s' unvollstaendig!", roiNameNumbered, badDepthParam);
								snprintf(str_en, sizeof(str_en), "ROI '%s': info parameter '%s' incomplete!", roiNameNumbered, badDepthParam);
							}
							else
							{
								snprintf(str_de, sizeof(str_de), "ROI '%s': Info-Parameter '%s' fehlt!", roiNameNumbered, badDepthParam);
								snprintf(str_en, sizeof(str_en), "ROI '%s': info parameter '%s' missing!", roiNameNumbered, badDepthParam);
							}
							SelectLanguage(RejectMessage, str_de, str_en);
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectMalfunction, Roi);
							break;
						}
						double threshold = 20;
						if (ArticleParameter(1) == ArticleColor_GNYE) threshold = 20;
						double x, y;
						AngleRad = DegToRad(90);
						
						MeasurePoint(hiPImg,// image
							Roi,			// region
							AngleRad,		// angle [rad]
							1.2,			    // sigma
							threshold,				// threshold
							"negative",		    // transition: all, negative, positive
							"first",		// select: all, first, last
							Paintings,		// paintings
							&x, &y);		// result
						//reference point -> Row2  to get spring depth
						double depthValue = 0;
						if (y != 0) depthValue = PixelToMm(DistancePP(PoleCenterX, PoleCenterY, PoleCenterX, y));
						if (InspectionActive(InspectionNo + 1) && (depthValue < MinValue || depthValue > MaxValue))
						{
							hiPImg = ScaleImageMinMax(hiPImg, 50, 100);
							MeasurePoint(hiPImg,// image
								Roi,			// region
								AngleRad,		// angle [rad]
								1.2,			    // sigma
								threshold,				// threshold
								"negative",		    // transition: all, negative, positive
								"first",		// select: all, first, last
								Paintings,		// paintings
								&x, &y);		// result
							if (y != 0) depthValue = PixelToMm(DistancePP(PoleCenterX, PoleCenterY, PoleCenterX, y));
							if (InspectionActive(InspectionNo + 1) && (depthValue < MinValue || depthValue > MaxValue))
							{
								snprintf(str_de, sizeof(str_de), "Springfläche%i: %.2f mm (Min %.2f, Max %.2f)", springNo, depthValue, MinValue, MaxValue);
								snprintf(str_en, sizeof(str_en), "Spring depth%i: %.2f mm (Min %.2f, Max %.2f)", springNo, depthValue, MinValue, MaxValue);
								SelectLanguage(RejectMessage, str_de, str_en);
								RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectSpringError, Roi);
								CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth);
								break;
							}
						}
						// set data chart
						MeasurementChartSet(MeasurementChartNo + 3, depthValue, MinValue, MaxValue, MinValue - 2.0, MaxValue + 2.0);//10

					}

					// paintings
					//CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 0, 200, 1);
					//CHalBase::PaintRegion(spring, WindowWithPaintings, R, G, B, 1);
					//CHalBase::PaintRegion(Roi, WindowAdditional1, 0, 0, 200, 1);
					//CHalBase::PaintRegion(spring, WindowAdditional1, R, G, B, 1);
					snprintf(str_de, sizeof(str_de), "Feder %i", springNo);
					snprintf(str_en, sizeof(str_en), "Spring %i", springNo);
					SelectLanguage(string1, str_de, str_en);
					CenterRow = Roi.SmallestRectangle2(&CenterCol, nullptr, &Length1, &Length2);
					CHalBase::PaintText(string1, CenterCol - Length2, CenterRow + Length1 + FontSize, WindowWithPaintings, R, G, B, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignRight, eAlignCenter, true);
					if (InspectionActive(Debug_MODE))
						CHalBase::PaintText(string1, CenterCol - Length2, CenterRow + Length1 + FontSize, WindowAdditional1, R, G, B, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignRight, eAlignCenter, true);
				}
			} while (!Roi.IsEmpty() && !RejectPole());
			
			// spring checked
			if (!checked)
			{
				snprintf(str_de, sizeof(str_de), "ROI für Federprüfung fehlt!");
				snprintf(str_en, sizeof(str_en), "ROI of spring check is missing!");
				SelectLanguage(RejectMessage, str_de, str_en);
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
			}
			Logger::getInstance().LogInfo("3.1 Spring missing/double stop", CameraNo);
			TimeMeasurementStop();
		}
#pragma endregion



		////////////////////////
		// 4.2 Pusher orientation
#pragma region
		InspectionNo			= 18;
		RejectStatisticGroupNo	=  4;
		RejectStatisticEntryNo	=  2;

		int InspectionNo_Underfilled = 19;
		int RejectStatisticEntryNo_Underfilled = 3;

		int InspectionNo_Tip = 21;
		int RejectStatisticEntryNo_Tip = 5;

		MeasurementChartNo		= 12;						// Score
		MeasurementChartNo2		= MeasurementChartNo + 1;	// Angle
		MeasurementChartNo3		= MeasurementChartNo2 + 1;	// Distance to target position
		MeasurementChartNo4		= MeasurementChartNo3 + 1;	// Underfilled
		Paintings				= PaintingsEditable2();

#ifdef _Halcon_ShapeModel
		if (!RejectPole() && InspectionActive(InspectionNo ))	// requires inspection 4.1 Pusher missing
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("4.2 Pusher orientation started", CameraNo);
			if (ShapeModelUsed)
			{
				// copy pusher region to pusher filter image
			    pusherImg = HImage::GenImageConst("byte", ImageWidth(), ImageHeight());
				for (size_t pusherNo = 0; pusherNo < article.PushersCount() && !RejectPole(); pusherNo++)
				{
					HRegion pusher = article.Pushers(pusherNo)->Region();
					pusherImg.OverpaintRegion(pusher, 255, "fill");
				}
				
				pusherImg = pusherImg.MeanImage(3, 3);
				//pusherImg.WriteImage("png", 0, "d:/pusherImg.png");
				if (InspectionActive(Debug_MODE)) CHalBase::HImage1ToRgbh(pusherImg, WindowAdditional2, CameraNo, 0);
			
				// find pusher in pusher filter image
				CHal_HImage *lastShapemodelImage = poSMs->HImg();
				poHImgPusher = new CHal_HImage(CameraNo, WindowAdditional2, WindowAdditional2);
				poHImgPusher->setImg(pusherImg);
				poSMs->SetHImg(poHImgPusher);
				for (size_t pusherNo = article.FirstPusher(); pusherNo < article.PushersCount() && !RejectPole(); pusherNo++)
				{
					// skip 'not inserted' pushers
					if (article.Pushers(pusherNo)->Type() == PusherTypes::NotInserted)
						continue;

					// get pusher roi
					snprintf(roiNameNumbered, sizeof(roiNameNumbered), "%s%i", Pusher::RoiName().c_str(), pusherNo + 1);
					poSMs->getRoiAndParams(oResHousing, roiNameNumbered, Roi);
					if (Roi.IsEmpty())
					{
						snprintf(str_de, sizeof(str_de), "ROI für Pusher %i fehlt!", pusherNo + 1);
						snprintf(str_en, sizeof(str_en), "ROI of Pusher %i is missing!", pusherNo + 1);
						SelectLanguage(RejectMessage, str_de, str_en);
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
						SetResultPusher(pusherNo + 1, !Reject());	// set pusher to reject
						break;
					}
					else
					{
						// use angle from pusher roi
						Roi.SmallestRectangle2(nullptr, &AngleRad, nullptr, nullptr);
						double PusherRoiAngle = AngleRad * 180 / g_PI;
						if (AngleRad < 0)
							AngleRad += g_PI;

						// put roi below straight pusher edge
						HRegion pusher = article.Pushers(pusherNo)->Region();
						CHalBase::PaintRegion(pusher, WindowWithPaintings, 255, 255, 0,3);
						
						// ROI angle vorbereiten und anzeigen
						Row1 = pusher.SmallestRectangle1(&Col1, &Row2, &Col2);
						double CorrectedROIAngle = PusherRoiAngle;
						if (CorrectedROIAngle < (-45)) CorrectedROIAngle = CorrectedROIAngle + 90;
						if (CorrectedROIAngle > 45) CorrectedROIAngle = CorrectedROIAngle - 90;
						snprintf(string1, sizeof(string1), "ROI angle = %.2f", CorrectedROIAngle);
						//CHalBase::PaintText(string1, Col1 + 60, Row1 + 40, WindowAdditional2, 200, 200, 200, (int)(0.60 * FontSize), HW_FONT_ARIAL, true, false, 0, eAlignLeft, eAlignCenter, true);

						if (fabs(PusherRoiAngle - 90.0) < numeric_limits<double>::epsilon())	// angle == 90.0
						{
							// straight pushers
							pusher.SmallestRectangle1(&Col1, &Row2, &Col2);
							CenterRow = Row2;
							if (article.Pushers(pusherNo)->Orientation() == PusherOrientations::Left)
								CenterCol = Col1;
							else if (article.Pushers(pusherNo)->Orientation() == PusherOrientations::Right)
								CenterCol = Col2;
							Roi = HRegion::GenCircle(CenterRow, CenterCol, MmToPixelDouble(1.00));
						}
						else
						{
							// rotated pushers
							CenterRow = pusher.SmallestRectangle2(&CenterCol, nullptr, &Length1, &Length2);	// problemes if the rectangle doesn't align properly on the left/right pusher side
							//CHalBase::PaintRectangle2(CenterCol, CenterRow, testPhi, Length1, Length2, WindowAdditional2, 200, 200, 0);
							if (article.Pushers(pusherNo)->Orientation() == PusherOrientations::Left)
								PointOnVector(CenterCol, CenterRow, AngleRad + g_PI_2, Length2, CenterCol, CenterRow);
							else if (article.Pushers(pusherNo)->Orientation() == PusherOrientations::Right)
								PointOnVector(CenterCol, CenterRow, AngleRad - g_PI_2, Length2, CenterCol, CenterRow);
							PointOnVector(CenterCol, CenterRow, AngleRad + g_PI, Length1, CenterCol, CenterRow);
							Roi = HRegion::GenCircle(CenterRow, CenterCol, MmToPixelDouble(1.00));
							CHalBase::PaintCross(CenterCol, CenterRow, Round_dti(Length1 * 2.0), WindowWithPaintings, 200, 200, 0);
						}

						// set minimum/maximum score
						MinValue = 0;
						if (InspectionActive(InspectionNo))
							MinValue = 85;

						// search pusher ShapeModel
						snprintf(roiName, sizeof(roiName), "%s", article.Pushers(pusherNo)->ShapeModelName().c_str());
						ShapeModelActive = roiName;
						poSMs->find(ShapeModelActive,	// name of ShapeModel
							Roi,						// search position X/Y, default=0/0 for using ShapeModel settings
							ShapeModelActivate,			// activated ShapeModel (0 for using ShapeModel settings)
							true);						// true:	ShapeModel should be found (MinValue used)
														// false:	ShapeModel should not be found (MaxValue used)

						// get ShapeModel results
						SMLibResult oRes;
						InitSMResult(&oRes);
						oRes = poSMs->res(ShapeModelActive,	// name of ShapeModel
							MeasurementChartNo,				// measurement chart number for score, default=-1
							MinValue, //MaxValue,			// minimum / maximum score
							Paintings,						// activate paintings, default=1
							1,								// width paintings, default=1
							0);								// index, default=0

						// ShapeModel was found
						//TRACE("%.2f°  ", oRes.angle_deg);
						if (!poSMs->Err(ShapeModelActive))
						{
							// save reference point as target pusher tip
							article.Pushers(pusherNo)->SetTargetTip(oRes.x_ref_point, oRes.y_ref_point);

							// check angle difference between roi and found shape
							MaxValue = 5;	// +/- deg
							MaxValue = 7;

							// check pusher shape angle
							double PusherAngle = oRes.angle_deg;
							if (PusherRoiAngle > 0)
								PusherRoiAngle = PusherRoiAngle - 90;
							else
								PusherRoiAngle = PusherRoiAngle + 90;

							Row1 = pusher.SmallestRectangle1(&Col1, &Row2, &Col2);
							snprintf(string1, sizeof(string1), "Shape angle = %.2f", oRes.angle_deg);
							CHalBase::PaintText(string1, Col1 + 60, Row1 + 55, WindowAdditional2, 200, 200, 200, (int)(0.60 * FontSize), HW_FONT_ARIAL, true, false, 0, eAlignLeft, eAlignCenter, true);

							double PusherAngleDifference = CorrectedROIAngle - oRes.angle_deg;
							snprintf(string1, sizeof(string1), "Angle diff.= %.2f", PusherAngleDifference);

							if (fabs(PusherAngleDifference) < MaxValue)
								CHalBase::PaintText(string1, Col1 + 60, Row1 + 70, WindowAdditional2, 200, 200, 200, (int)(0.60 * FontSize), HW_FONT_ARIAL, true, false, 0, eAlignLeft, eAlignCenter, true);
							else
								CHalBase::PaintText(string1, Col1 + 60, Row1 + 70, WindowAdditional2, 200, 0, 0, (int)(0.60 * FontSize), HW_FONT_ARIAL, true, false, 0, eAlignLeft, eAlignCenter, true);

							Value = abs(PusherAngle - PusherRoiAngle);
							if (InspectionActive(InspectionNo) && Value > MaxValue)
							{
								snprintf(str_de, sizeof(str_de), "Pusherlage: Shapemodel angle: %.2f deg (Max %.2f)", Value, MaxValue);
								snprintf(str_en, sizeof(str_en), "Pusher orientation: Shapemodel angle: %.2f deg (Max %.2f)", Value, MaxValue);
								SelectLanguage(RejectMessage, str_de, str_en);
								RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPusherError, Roi);
								SetResultPusher(pusherNo + 1, !Reject());	// set pusher to reject
							}
							// set data chart
							MeasurementChartSet(MeasurementChartNo2, Value, Hidden, MaxValue);

							// also check distance of reference point to center of roi
							MaxValue = 0.32;
							MaxValue = 0.40;	// 2023-11-03 to start production
							if (ArticleParameter(1) == ArticleColor_YE)MaxValue = 0.55;
							Value = PixelToMm(DistancePP(CenterCol, CenterRow, oRes.x_ref_point, oRes.y_ref_point));
							if (InspectionActive(InspectionNo) && Value > MaxValue)
							{
								snprintf(str_de, sizeof(str_de), "Pusherlage: Abstand zur Sollposition: %.2f mm (Max %.2f)", Value, MaxValue);
								snprintf(str_en, sizeof(str_en), "Pusher orientation: Distance to target position: %.2f mm (Max %.2f)", Value, MaxValue);
								SelectLanguage(RejectMessage, str_de, str_en);
								RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPusherError, Roi);

								// Distance not OK: show distance with arrow
								CHalBase::PaintArrow2(CenterCol, CenterRow, oRes.x_ref_point, oRes.y_ref_point, 5, 5, WindowWithPaintings, R, G, B, 1);
								SetResultPusher(pusherNo + 1, !Reject());	// set pusher to reject
							}

							// set data chart
							MeasurementChartSet(MeasurementChartNo3, Value, Hidden, MaxValue);

							// paintings. If reject then paint always.
							if ((PaintingsEditable2() > 1) || Reject())
							{
								CHalBase::PaintCircle(CenterCol, CenterRow, MmToPixel(MaxValue), WindowAdditional2, R, G, B);
								CHalBase::PaintLine(CenterCol, CenterRow, oRes.x_ref_point, oRes.y_ref_point, WindowAdditional2, R, G, B);

								// Paint reference point of each pusher
								CHalBase::PaintCross(oRes.x_ref_point, oRes.y_ref_point, 10, WindowWithPaintings, 0, 255, 255, LineWidth + 1);
							}
						}
						else
						{
							if (InspectionActive(InspectionNo))
								RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, poSMs->getErr(ShapeModelActive), PlcRejectPusherError);				// Lage prüfen ==> Fehler Lage
							else
							{
								if (InspectionActive(InspectionNo_Underfilled))
								{
									RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo_Underfilled, poSMs->getErr(ShapeModelActive), PlcRejectPusherError);	// Lage nicht prüfen ==> Fehler Unterspritzung
								}
								else
								{
									if (InspectionActive(InspectionNo_Tip))
									{
										RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo_Tip, poSMs->getErr(ShapeModelActive), PlcRejectPusherError);		// Lage nicht prüfen ==> Fehler Unterspritzung
									}
								}
							}
							SetResultPusher(pusherNo + 1, !Reject());	// set pusher to reject
						}

						// paintings
						snprintf(str_de, sizeof(str_de), "Pusher %i", pusherNo + 1);
						snprintf(str_en, sizeof(str_en), "Pusher %i", pusherNo + 1);
						SelectLanguage(string1, str_de, str_en);
						CenterRow = pusher.SmallestRectangle2(&CenterCol, nullptr, &Length1, &Length2);
						if (PaintingsEditable2())
						{
							CHalBase::PaintText(string1, CenterCol - Length2 * 2, CenterRow - Length1, WindowWithPaintings, R, G, B, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignRight, eAlignCenter, true);
							SetResultPusher(pusherNo + 1, !Reject());	// set pusher to reject

							// Enhanced shape model result paintings (scoreResult)
							double dX1, dY1; Roi.AreaCenter(&dY1, &dX1); X1 = (int)dX1; Y1 = (int)dY1;
							EnhancedModelResultPaintings(oRes, Round_dti(CenterCol) - MmToPixel(4.00), Round_dti(CenterRow) + MmToPixel(1.50),/* X1, Y1,*/ false, MinValue, WindowAdditional2, 100, R, G, B);
						}

						// break on first error
						if (RejectPole())
						{
							//CHalBase::PaintText(string1, CenterCol - Length2 * 2, CenterRow - Length1, WindowWithPaintings, R, G, B, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignRight, eAlignCenter, true);
							SetResultPusher(pusherNo + 1, !Reject());	// set pusher to reject
							break;
						}



						////////////////////////
						// 4.3 Pusher underfilled
						else if (InspectionActive(InspectionNo + 1))
						{
							snprintf(roiName, sizeof(roiName), "%s", Pusher::RoiUnderfilledName().c_str());
							poSMs->getRoiAndParams(oRes, roiName, Roi);
							if (Roi.IsEmpty())
							{
								snprintf(str_de, sizeof(str_de), "%i.%i ROI für 'Pusher unterspritzt' fehlt!", RejectStatisticGroupNo, RejectStatisticEntryNo + 1);
								snprintf(str_en, sizeof(str_en), "%i.%i ROI of 'Pusher underfilled' is missing!", RejectStatisticGroupNo, RejectStatisticEntryNo + 1);
								SelectLanguage(RejectMessage, str_de, str_en);
								RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
								SetResultPusher(pusherNo + 1, !Reject());	// set pusher to reject
								break;
							}
							else
							{
								if (PaintingsEditable2())
									CHalBase::PaintRegion(Roi, WindowAdditional2, 0, 0, 200);

								Roi = Roi.Intersection(pusher);
								MinValue = article.Pushers(pusherNo)->MinUnderfilledSize();
								MaxValue = article.Pushers(pusherNo)->MaxUnderfilledSize();
								Value = PixelToMm2(Roi.Area());

								// check pusher underfilled size
								if (Value < MinValue || Value > MaxValue)
								{
									snprintf(str_de, sizeof(str_de), "Pusherfläche: %.2f mm^2 (Min %.2f, Max %.2f)", Value, MinValue, MaxValue);
									snprintf(str_en, sizeof(str_en), "Pusher area: %.2f mm^2 (Min %.2f, Max %.2f)", Value, MinValue, MaxValue);
									SelectLanguage(RejectMessage, str_de, str_en);
									RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectPusherError, Roi);
									SetResultPusher(pusherNo + 1, !Reject());	// set pusher to reject
								}

								// set data chart
								MeasurementChartSet(MeasurementChartNo4, Value, MinValue, MaxValue, MinValue - 0.5, MaxValue + 0.5);

								// paintings. If reject then paint always.
								if (PaintingsEditable2() || Reject())
									CHalBase::PaintRegionFilled(Roi, WindowAdditional2, R, G, B);
							}
						}
					}
				}
				//TRACE("\n");
				poSMs->SetHImg(lastShapemodelImage);	// switch shape model image back to camera image
			}
			Logger::getInstance().LogInfo("4.2 Pusher orientation stop", CameraNo);
			TimeMeasurementStop();
		}
#endif
#pragma endregion



		////////////////////////
		// 4.4 Pusher depth
#pragma region
		InspectionNo			= 20;
		RejectStatisticGroupNo	= 4;
		RejectStatisticEntryNo	= 4;

		InspectionNo_Tip			= 21;
		RejectStatisticEntryNo_Tip	= 5;

		MeasurementChartNo		= 16;							// Depth
		MeasurementChartNo2		= MeasurementChartNo + 1;		// Tip
		Paintings				= PaintingsEditable2();
	    
		if (!RejectPole() && (InspectionActive(InspectionNo) ))
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("4.4 Pusher depth started", CameraNo);
			if (!InspectionActive(InspectionNo - 2))
			{
			    pusherImg = HImage::GenImageConst("byte", ImageWidth(), ImageHeight());
				for (size_t pusherNo = 0; pusherNo < article.PushersCount() && !RejectPole(); pusherNo++)
				{
					HRegion pusher = article.Pushers(pusherNo)->Region();
					pusherImg.OverpaintRegion(pusher, 255, "fill");
				}

				pusherImg = pusherImg.MeanImage(3, 3);
				if (InspectionActive(Debug_MODE)) CHalBase::HImage1ToRgbh(pusherImg, WindowAdditional2, CameraNo, 0);
			}
			
			snprintf(str_de, sizeof(str_de), "%i.%i Eindrücktiefe Pusher", RejectStatisticGroupNo, RejectStatisticEntryNo);
			snprintf(str_en, sizeof(str_en), "%i.%i Pusher depth", RejectStatisticGroupNo, RejectStatisticEntryNo);
			SelectLanguage(InspectionName, str_de, str_en);

			for (size_t pusherNo = article.FirstPusher(); pusherNo < article.PushersCount() && !RejectPole(); pusherNo++)
			{
				// skip 'not inserted' pushers
				if (article.Pushers(pusherNo)->Type() == PusherTypes::NotInserted)
					continue;

				// get pusher ROI
				snprintf(roiNameNumbered, sizeof(roiNameNumbered), "%s%i", Pusher::RoiName().c_str(), pusherNo + 1);
				poSMs->getRoiAndParams(oResHousing, roiNameNumbered, Roi);
				if (Roi.IsEmpty())
				{
					snprintf(str_de, sizeof(str_de), "ROI für Pusher %i fehlt!", pusherNo + 1);
					snprintf(str_en, sizeof(str_en), "ROI of pusher %i is missing!", pusherNo + 1);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
					break;
				}
				else
				{
					// get angle to turn pusher straight
					Roi.SmallestRectangle2(nullptr, &AngleRad, nullptr, nullptr);
					double turnAngleRad = Abs(AngleRad - g_PI_2);
					if (AngleRad < 0)
						turnAngleRad = Abs(AngleRad) - g_PI_2;

					// get smallest rectangle of straight pusher
					HRegion pusher = article.Pushers(pusherNo)->Region();

					Row1 = pusher.SmallestRectangle1(&Col1, &Row2, &Col2);	// don't use SmallestRectangle2, it can align to the wrong pusher side
					CenterCol = Col1 + 0.5 * (Col2 - Col1);
					CenterRow = Row1 + 0.5 * (Row2 - Row1);
					CHalBase::PaintCross(CenterCol, CenterRow,10,WindowWithPaintings,0,255,0,5);
					HTuple HomMat2D;
					hom_mat2d_identity(&HomMat2D);
					hom_mat2d_rotate(HomMat2D, turnAngleRad, CenterRow, CenterCol, &HomMat2D);
					//CHalBase::PaintRegionFilled(pusher.AffineTransRegion(HomMat2D, "false"), WindowWithPaintings, 250, 250, 250);
	
					Row1 = pusher.AffineTransRegion(HomMat2D, "false").SmallestRectangle1(&Col1, &Row2, &Col2);
					// get pusher region size
					double halfWidth = (Col2 - Col1) / 2.0;
					double halfHeight = ((Row2 - Row1) / 2.0);// +MmToPixel(0.50);
					double topWidthOffset = 0;
					if (AngleRad < 0)
						AngleRad += g_PI;

					if (halfWidth && halfHeight)	// Stop pusher checks, if region is empty
					{
						if (article.HasLShapePusher()) halfHeight = halfHeight + MmToPixel(0.50);
						halfWidth = halfWidth + MmToPixel(0.50);
						// measure pusher top in filter image
						R = B = 0; G = 200;
						HTuple colEdge, rowEdge, amplitude;
						HRegion roi;

						// top roi position: move center point to pusher upper line(CenterRow--)
						PointOnVector(CenterCol, CenterRow, AngleRad + g_PI_2, MmToPixelDouble(article.Pushers(pusherNo)->OffsetXTopDepth()), CenterCol, CenterRow);//move X
						PointOnVector(CenterCol, CenterRow, AngleRad , MmToPixelDouble(article.Pushers(pusherNo)->OffsetXBottomDepth()), CenterCol, CenterRow);//move Y
						if (article.HasLShapePusher())
							topWidthOffset = -MmToPixelDouble(0.65);
						else
							topWidthOffset = 0;
						roi = HRegion::GenRectangle2(CenterRow, CenterCol, AngleRad, halfHeight/1.2, halfWidth  + topWidthOffset);
						CHalBase::PaintRegion(roi, WindowAdditional2, 255, 255, 0, 1);
						
						/*HImage testimg = poHImgPusher->getImg()->AccessChannel(1);
						testimg.WriteImage("png",0,"d:/test.png");*/
						HMeasure measureTop(CenterRow, CenterCol, AngleRad, halfHeight/1.2, halfHeight + topWidthOffset, ImageWidth(), ImageHeight(), "nearest_neighbor");
						double thre = 30;
						if (ArticleParameter(1) == ArticleColor_GNYE) thre = 20;
						rowEdge = measureTop.MeasurePos(/*poHImgPusher->getImg()->AccessChannel(1)*/pusherImg,
							2.0,			// Sigma: default=1.0
							10,				// Threshold: default=30
							"negative",		// Transition: 'all', 'negative' (hell->dunkel), 'positive' (dunkel->hell)
							"all",			// Select: 'all', 'first', 'last'
							&colEdge,
							&amplitude,
							nullptr);

						if (amplitude.Num() <= 0)
						{
							snprintf(str_de, sizeof(str_de), "Obere Gehäusekante an Pusher %i nicht gefunden!", pusherNo + 1);
							snprintf(str_en, sizeof(str_en), "Top Housing edge at pusher %i not found!", pusherNo + 1);
							SelectLanguage(RejectMessage, str_de, str_en);
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPusherWrongDepth, roi);
							CHalBase::PaintRegion(roi, WindowWithPaintings, R, G, B, LineWidth);
							break;
						}
						else
						{
							// use edge with highest amplitude
							int index = 0;
							if (amplitude.Num() > 1)
								index = amplitude.Abs().SortIndex().Inverse()[0].I();
							double topX = colEdge[index].D();
							double topY = rowEdge[index].D();
							//Trace("%i: %.2f\n", pusherNo, topY);
							if (PaintingsEditable2())
								CHalBase::PaintCrossAngle(topX, topY, 5, AngleRad, WindowAdditional2, 0, 0, 200);

							
							//PointOnVector(topX, topY, AngleRad + g_PI_2, MmToPixelDouble(article.Pushers(pusherNo)->OffsetXTopDepth()), topX, topY);
							// shift back 
							PointOnVector(CenterCol, CenterRow, -AngleRad, MmToPixelDouble(article.Pushers(pusherNo)->OffsetXBottomDepth()), CenterCol, CenterRow);//shift back Y
							//PointOnVector(CenterCol, CenterRow, g_PI -(AngleRad + g_PI_2), MmToPixelDouble(article.Pushers(pusherNo)->OffsetXTopDepth()), CenterCol, CenterRow);//shift back x

							// measure pusher bottom in filter image
							// use bottom offset to shift region to deepest point of the pusher
							//PointOnVector(CenterCol, CenterRow, AngleRad + g_PI_2, MmToPixelDouble(article.Pushers(pusherNo)->OffsetXTopDepth()), CenterCol, CenterRow);//move x
							PointOnVector(CenterCol, CenterRow, -AngleRad, MmToPixelDouble(article.Pushers(pusherNo)->OffsetXBottomDepth()), CenterCol, CenterRow);//move Y
							roi = HRegion::GenRectangle2(CenterRow, CenterCol, AngleRad, halfHeight/1.2, halfWidth);
							CHalBase::PaintRegion(roi, WindowAdditional2, 255, 255, 0, 1);
						
							HMeasure measureBottom(CenterRow, CenterCol, AngleRad, halfHeight/1.2, halfWidth, ImageWidth(), ImageHeight(), "nearest_neighbor");
							rowEdge = measureBottom.MeasurePos(/*poHImgPusher->getImg()->AccessChannel(1)*/pusherImg,
								2.0,			// Sigma: default=1.0
								30,				// Threshold: default=30
								"positive",		// Transition: 'all', 'negative' (hell->dunkel), 'positive' (dunkel->hell)
								"all",			// Select: 'all', 'first', 'last'
								&colEdge,
								&amplitude,
								nullptr);

							if (amplitude.Num() <= 0)
							{
								snprintf(str_de, sizeof(str_de), "Untere Gehäusekante an Pusher %i nicht gefunden!", pusherNo + 1);
								snprintf(str_en, sizeof(str_en), "Bottom housing edge at pusher %i not found!", pusherNo + 1);
								SelectLanguage(RejectMessage, str_de, str_en);
								RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPusherWrongDepth, roi);
								CHalBase::PaintRegion(roi, WindowWithPaintings, R, G, B, LineWidth);
								break;
							}
							else
							{
								// use edge with highest amplitude
								index = 0;

								if (rowEdge.Num() > 1)
									index = rowEdge.SortIndex().Inverse()[0].I();
								double bottomX = colEdge[index].D();
								double bottomY = rowEdge[index].D();
								if (PaintingsEditable2())
									CHalBase::PaintCrossAngle(bottomX, bottomY, 5, AngleRad, WindowAdditional2, 0, 0, 250);

								// measure distance from pusher top to pusher bottom
								//PointOnVector(bottomX, bottomY, AngleRad + g_PI_2, MmToPixelDouble(article.Pushers(pusherNo)->OffsetXBottomDepth()), bottomX, bottomY);	// shift bottom result back if it was shifted
								distance_pp(topY, topX, bottomY, topX, &Value);
								MinValue = article.Pushers(pusherNo)->MinDepth();
								MaxValue = article.Pushers(pusherNo)->MaxDepth();
								Value = PixelToMm(Value);

								// check pusher depth
								if (InspectionActive(InspectionNo))
								{
									if (Value < MinValue || Value > MaxValue)
									{
										snprintf(str_de, sizeof(str_de), "Eindrücktiefe Pusher %i: %.3fmm (Min %.2f, Max %.2f)", pusherNo + 1, Value, MinValue, MaxValue);
										snprintf(str_en, sizeof(str_en), "Pusher %i depth: %.2fmm (Min %.3f, Max %.2f)", pusherNo + 1, Value, MinValue, MaxValue);
										SelectLanguage(RejectMessage, str_de, str_en);
										RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPusherWrongDepth, Roi);

										// overpaint pusher text
										snprintf(str_de, sizeof(str_de), "Pusher %i", pusherNo + 1);
										snprintf(str_en, sizeof(str_en), "Pusher %i", pusherNo + 1);
										SelectLanguage(string1, str_de, str_en);
										CenterRow = pusher.SmallestRectangle2(&CenterCol, nullptr, &Length1, &Length2);
										//CHalBase::PaintText(string1, CenterCol - Length2 * 2, CenterRow - Length1, WindowWithPaintings, R, G, B, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignRight, eAlignCenter, true);
										SetResultPusher(pusherNo + 1, !Reject());	// set pusher to reject
									}

									// set data chart
									MeasurementChartSet(MeasurementChartNo, Value, MinValue, MaxValue, MinValue - 0.20, MaxValue + 0.20);

									// paintings
									double x1, x2, y1, y2;//, minX, minY, maxX, maxY;
									PointOnVector(topX, topY, AngleRad - g_PI_2, halfWidth, x1, y1);
									PointOnVector(topX, topY, AngleRad + g_PI_2, halfWidth, x2, y2);
									CHalBase::PaintLine(x1, y1, x2, y2, WindowWithPaintings, 0, 0, 200, 5);		// top housing edge
									PointOnVector(bottomX, bottomY, AngleRad - g_PI_2, halfWidth, x1, y1);
									PointOnVector(bottomX, bottomY, AngleRad + g_PI_2, halfWidth, x2, y2);
									CHalBase::PaintLine(x1, y1, x2, y2, WindowWithPaintings, 0, 200, 200, 5);	// bottom pusher edge
									//PointOnVector(topX, topY, AngleRad - g_PI, MmToPixel(MinValue), minX, minY);
									//PointOnVector(minX, minY, AngleRad - g_PI_2, halfWidth, x1, y1);
									//PointOnVector(minX, minY, AngleRad + g_PI_2, halfWidth, x2, y2);
									//CHalBase::PaintLine(x1, y1, x2, y2, WindowWithPaintings, 0, 250, 0, 1);		// min pusher depth
									//PointOnVector(topX, topY, AngleRad - g_PI, MmToPixel(MaxValue), maxX, maxY);
									//PointOnVector(maxX, maxY, AngleRad - g_PI_2, halfWidth, x1, y1);
									//PointOnVector(maxX, maxY, AngleRad + g_PI_2, halfWidth, x2, y2);
									//CHalBase::PaintLine(x1, y1, x2, y2, WindowWithPaintings, 0, 250, 0, 1);		// max pusher depth
									CHalBase::PaintArrow2(topX, topY, bottomX, bottomY, 5, 5, WindowWithPaintings, R, G, B, 5);
								}



								//////////////////////////
								//// 4.5 Pusher tip visible (only LShape)
								//if (!RejectPole() && InspectionActive(InspectionNo_Tip) && article.Pushers(pusherNo)->Type() == PusherTypes::LShape)
								//{
								//	PointD targetTip = article.Pushers(pusherNo)->TargetTip();
								//	double distance = DistancePP(targetTip.X, targetTip.Y, bottomX, bottomY);	// max possible distance of reference point and pusher bottom
								//	double x, y;
								//	PointOnVector(targetTip.X, targetTip.Y, AngleRad, distance + 1, x, y);		// to get a straight from pusher reference point to pusher top
								//	ProjectionPL(bottomX, bottomY, x, y, targetTip.X, targetTip.Y, x, y);
								//	MaxValue = 0.40;
								//	Value = PixelToMm(DistancePP(x, y, targetTip.X, targetTip.Y));

								//	// check pusher tip distance to pusher tip reference point
								//	if (Value > MaxValue)
								//	{
								//		snprintf(str_de, sizeof(str_de), "Pusherspitze nicht gefunden: Abstand zur Sollposition %.2f mm (Max %.2f)", Value, MaxValue);
								//		snprintf(str_en, sizeof(str_en), "Pusher tip not found: Distance to target position %.2f mm (Max %.2f)", Value, MaxValue);
								//		SelectLanguage(RejectMessage, str_de, str_en);
								//		RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectPusher, HRegion::GenCircle(targetTip.Y, targetTip.X, distance));
								//	}

								//	// paintings
								//	CHalBase::PaintCrossAngle(targetTip.X, targetTip.Y, 5, RadToDeg(AngleRad), WindowWithPaintings, 0, 250, 250);
								//	CHalBase::PaintCrossAngle(x, y, 5, RadToDeg(AngleRad), WindowWithPaintings, R, G, B);
								//	if (Paintings > 0)
								//	{
								//		CHalBase::PaintCrossAngle(targetTip.X, targetTip.Y, 5, RadToDeg(AngleRad), WindowAdditional2, 0, 250, 250);
								//		CHalBase::PaintCrossAngle(x, y, 5, RadToDeg(AngleRad), WindowAdditional2, R, G, B);
								//	}

								//	// set data chart
								//	MeasurementChartSet(MeasurementChartNo2, Value, Hidden, MaxValue);
								//}
							}
						}
					}
					else
					{
						snprintf(str_de, sizeof(str_de), "Pusher %i: Keine Region vorhanden", pusherNo + 1);
						snprintf(str_en, sizeof(str_en), "Pusher %i: No region found", pusherNo + 1);
						SelectLanguage(RejectMessage, str_de, str_en);
						if (InspectionActive(InspectionNo))
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPusherError);
						if (InspectionActive(InspectionNo_Tip))
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo_Tip, RejectMessage, PlcRejectPusherError);
						SetResultPusher(pusherNo + 1, !Reject());	// set pusher to reject
					}
				}
			}
			Logger::getInstance().LogInfo("4.4 Pusher depth stop", CameraNo);
			TimeMeasurementStop();
		}
#pragma endregion
	

		//5.1 Slider check
		InspectionNo = 41;
		RejectStatisticGroupNo = 5;
		RejectStatisticEntryNo = 1;
		MeasurementChartNo = 28;
		Paintings = PaintingsEditable4();
		if (!RejectPole() && InspectionActive(InspectionNo))
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("5.1 slider check started", CameraNo);
			HTuple roiIndex, infoParams;
			const char* sliderName;
			articleDef.FindRegionNum(roiIndex, oResHousing, "Slider");
			
			for (int i = 0; i < roiIndex.Num(); i++)
			{
				Logger::getInstance().LogInfo("5.1 loop times  = " + std::to_string(i), CameraNo);
				sliderName = articleDef.GetRoiByFuzzyRegion(roiIndex, oResHousing, i, Roi, poSMs, infoParams);


				const char* keys[] = { "", "emptySize", "minSize", "maxSize", "maxAngle" };
				int idxList[] = { 0, 1, 2, 3, 4 };
				int keyCnt = sizeof(keys) / sizeof(keys[0]);
				bool paramMiss = true;
				bool segEmpty = false;
				bool paramIncomplete = false;
				const char* badParam = "Info";
				HTuple parts;

				if (infoParams.Num() > 0 && infoParams.IsString())
				{
					HTuple str = infoParams[0];
					parts = str.Split(";");
					int partCount = parts.Num();
					paramMiss = false;

					for (int k = 0; !paramMiss && k < keyCnt; k++)
					{
						if (partCount <= idxList[k])
						{
							paramMiss = true;
							badParam = (k == 0) ? "L/R" : keys[k];
							break;
						}

						HTuple part = parts[idxList[k]];
						if (part.Strlen() == 0)
						{
							segEmpty = true;
							badParam = (k == 0) ? "L/R" : keys[k];
							break;
						}

						HTuple seg = part.Split(",");
						bool keyOk = false;
						if (k == 0)
						{
							keyOk = (seg.Num() >= 1 && (seg[0] == HTuple("L") || seg[0] == HTuple("R")));
						}
						else
						{
							string keyWithSpaceText = string(" ") + keys[k];
							HTuple keyWithSpace = HTuple(keyWithSpaceText.c_str());
							keyOk = (seg.Num() >= 1 && (seg[0] == HTuple(keys[k]) || seg[0] == keyWithSpace));
						}

						if (!keyOk)
						{
							paramMiss = true;
							badParam = (k == 0) ? "L/R" : keys[k];
							break;
						}

						int valueIndex = (k == 0) ? 0 : 1;
						if (seg.Num() <= valueIndex || HTuple(seg[valueIndex]).Strlen() == 0)
						{
							paramIncomplete = true;
							badParam = (k == 0) ? "L/R" : keys[k];
							break;
						}
					}
				}

				if (paramMiss || segEmpty || paramIncomplete)
				{
					char str_de[256] = { 0 }, str_en[256] = { 0 };
					if (paramIncomplete)
					{
						snprintf(str_de, sizeof(str_de), "ROI '%s': Info-Parameter '%s' unvollstaendig!", sliderName, badParam);
						snprintf(str_en, sizeof(str_en), "ROI '%s': info parameter '%s' incomplete!", sliderName, badParam);
					}
					else if (paramMiss)
					{
						snprintf(str_de, sizeof(str_de), "ROI '%s': Info-Parameter '%s' fehlt!", sliderName, badParam);
						snprintf(str_en, sizeof(str_en), "ROI '%s': info parameter '%s' missing!", sliderName, badParam);
					}
					else if (segEmpty)
					{
						snprintf(str_de, sizeof(str_de), "ROI '%s': Info-Parameter '%s' leer!", sliderName, badParam);
						snprintf(str_en, sizeof(str_en), "ROI '%s': info parameter '%s' empty!", sliderName, badParam);
					}
					else
					{
						snprintf(str_de, sizeof(str_de), "ROI '%s' nicht definiert!", sliderName);
						snprintf(str_en, sizeof(str_en), "ROI '%s' isn't defined!", sliderName);
					}
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, Roi);
					continue;
				}

				HTuple sliderType;
				double emptySize = 0.0;
				double minSize = 0.0;
				double maxSize = 0.0;
				double maxAngle = 0.0;

				for (int k = 0; k < keyCnt; k++)
				{
					HTuple seg = HTuple(parts[idxList[k]]).Split(",");
					HTuple value = (k == 0) ? seg[0] : seg[1];
					DumpTuple(value);

					switch (k)
					{
					case 0: sliderType = seg; break;
					case 1: emptySize = seg[1].D(); break;
					case 2: minSize = seg[1].D(); break;
					case 3: maxSize = seg[1].D(); break;
					case 4: maxAngle = seg[1].D(); break;
					}
				}

				//DumpTuple(infoParams);
				//HTuple str = infoParams[0];
				//HTuple parts = str.Split(";"); 
				////DumpTuple(parts);
				//if (strstr(infoParams[0], "L") > 0)
				//{
				//	HTuple params0 = HTuple(parts[0]).Split(",");
				//	DumpTuple(params0[0]);
				//}
				//if (strstr(infoParams[0], "emptySize") > 0)
				//{
				//	HTuple params0 = HTuple(parts[1]).Split(",");
				//	DumpTuple(params0[1]);						
				//}					
				//if (strstr(infoParams[0], "minSize") > 0)
				//{
				//	HTuple params0 = HTuple(parts[2]).Split(",");
				//	DumpTuple(params0[1]);
				//}
				//if (strstr(infoParams[0], "maxSize") > 0)
				//{
				//	HTuple params0 = HTuple(parts[3]).Split(",");
				//	DumpTuple(params0[1]);
				//}
				//if (strstr(infoParams[0], "maxAngle") > 0)
				//{
				//	HTuple params0 = HTuple(parts[4]).Split(",");
				//	DumpTuple(params0[1]);
				//}
			
					
					//DumpTuple(roiInfoParams);
					// check roi
					if (Roi.IsEmpty())
					{
						snprintf(str_de, sizeof(str_de), "ROI '%s' nicht definiert!", sliderName);
						snprintf(str_en, sizeof(str_en), "ROI '%s' isn't defined!", sliderName);
						SelectLanguage(RejectMessage, str_de, str_en);
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, Roi);
					}
					else
					{
						Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);
						Height = Abs(Row2 - Row1);
						Width = Abs(Col2 - Col1);
						X1 = (Col2 + Col1) / 2;
						Y1 = (Row2 + Row1) / 2;
						if (sliderType.Num() > 0 && sliderType[0] == HTuple("L")) ShapeModelActive = SM_SliderL;
						else ShapeModelActive = SM_SliderR;

						Logger::getInstance().LogInfo("5.1 loop times  = " + std::to_string(i) + ",find shap model start", CameraNo);
						int numMatches = 0;
						if (ArticleParameter(1) == ArticleColor_WH )
						{
							HImage originalImage = *poHImg->getImg();
							originalImage = ScaleImageMinMax(originalImage, 50, 200).Rgb1ToGray();
							//originalImage.WriteImage("jpg", 0, "D://1.JPG");
							numMatches = poSMs->find(originalImage,ShapeModelActive,	// name of ShapeModel
								X1, Y1,						// search position X/Y, default=0/0 for using ShapeModel settings
								Width, Height,				// search position width/height, default=0/0 for using ShapeModel settings
								ShapeModelActivate,			// activate ShapeModel (0 for using ShapeModel settings)
								true);						// true:	ShapeModel should be found (MinValue used)
						}
						else
						{

							numMatches = poSMs->find(ShapeModelActive,	// name of ShapeModel
								X1, Y1,						// search position X/Y, default=0/0 for using ShapeModel settings
								Width, Height,				// search position width/height, default=0/0 for using ShapeModel settings
								ShapeModelActivate,			// activate ShapeModel (0 for using ShapeModel settings)
								true);						// true:	ShapeModel should be found (MinValue used)
						}					// false:	ShapeModel should not be found (MaxValue used)

						// get ShapeModel results
						oResSlider = poSMs->res(ShapeModelActive,		// name of ShapeModel
							MeasurementChartNo,			// measurement chart number for score, default=-1
							MinValue, //MaxValue,		// minimum / maximum score
							Paintings,					// activate paintings, default=1
							LineWidth,					// width paintings, default=1
							0);
						Logger::getInstance().LogInfo("5.1 loop times  = " + std::to_string(i) + ",find shap model stop", CameraNo);
						// ShapeModel was found
						if (!poSMs->Err(ShapeModelActive))
						{
							HTuple roiIndex;
							const char* regionName;
							double maxEmpty = emptySize;
							HRegion roiSilder;
							articleDef.FindRegionNum(roiIndex, oResSlider, "Burr");
							regionName = articleDef.GetRoiByFuzzyRegion(roiIndex, oResSlider, 0, roiSilder);
							CHalBase::PaintRegion(roiSilder, WindowWithPaintings, 0, 255, 0, LineWidth);
							HImage hImg = poHImg->getImg()->ReduceDomain(roiSilder).Rgb1ToGray();
							double sliderIntensity, sliderDeviation;
							sliderIntensity = roiSilder.Intensity(hImg, &sliderDeviation);
							if (sliderDeviation > maxEmpty)
							{
								snprintf(str_de, sizeof(str_de), "Sliderfläche: %.2f  ( Max %.2f)", sliderDeviation, maxEmpty);
								snprintf(str_en, sizeof(str_en), "Slider burr: %.2f  (M Max %.2f)", sliderDeviation, maxEmpty);
								SelectLanguage(RejectMessage, str_de, str_en);
								RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 2, RejectMessage, PlcRejectSliderError, roiSilder);
								CHalBase::PaintRegion(roiSilder, WindowWithPaintings, R, G, B, LineWidth);
							}


							MaxValue = maxAngle;
							// apply found position
							sliderAngle = oResSlider.angle_deg;
							// 5.3 check angle 
							if (InspectionActive(InspectionNo + 1) && abs(sliderAngle) > MaxValue)
							{
								snprintf(str_de, sizeof(str_de), "Sliderfläche: %.2f ° ( Max %.2f)", sliderAngle, MaxValue);
								snprintf(str_en, sizeof(str_en), "Slider angle: %.2f ° (M Max %.2f)", sliderAngle, MaxValue);
								SelectLanguage(RejectMessage, str_de, str_en);
								RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 2, RejectMessage, PlcRejectSliderError, Roi);
								CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth);
								break;
							}

						}
						else // ShapeModel was not found
						{
							// set reject  5.1
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, poSMs->getErr(ShapeModelActive), PlcRejectSliderError, poSMs->getSearchRegion(ShapeModelActive));
							break;
						}
						Logger::getInstance().LogInfo("5.1 loop times  = " + std::to_string(i) + ",shapemodel judge", CameraNo);
						// set data chart
						MeasurementChartSet(MeasurementChartNo + 1, Value, Hidden, MaxValue, Hidden, MaxValue + 2.0);//29
					}
					if (Paintings)
					{
						CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 0, 0, LineWidth * 2); // shadow
						CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth); // automatically red painted in case of reject
					}
					Logger::getInstance().LogInfo("5.2 start", CameraNo);
					// 5.2
					// check roi
					HRegion slider = HRegion::GenEmptyRegion(); 
					Pusher pusher;
					if (ArticleParameter(1) != ArticleColor_OG)
					{
						if (ArticleParameter(1) == ArticleColor_GNYE)
						{
							slider = pusher.FilterHSVForSlider(poHImg->getImg(), Roi, 5, 40, 40.0, 100.0, 40.0, 150.0, false);
						}
						else if (ArticleParameter(1) == ArticleColor_BK)
						{
							slider = pusher.FilterHSVForSlider(poHImg->getImg(), Roi, 5, 40, 40.0, 100.0, 50.0, 150.0, false);
						}
						else if (ArticleParameter(1) == ArticleColor_RD)
						{
							slider = pusher.FilterHSVForSlider(poHImg->getImg(), Roi, 5, 40, 40.0, 100.0, 50.0, 150.0, false);
						}
						else slider = pusher.FilterHSVForSlider(poHImg->getImg(), Roi, 5, 40, 40.0, 100.0, 40.0, 150.0, false);//5, 40, 40.0, 100.0, 50.0, 150.0, false
						//slider = pusher.FilterHSVForSlider(poHImg->getImg(), Roi, 5, 40, 40.0, 100.0, 65.0, 150.0, false);
						slider = SortRegionArea(slider.Connection(), true)[0].OpeningCircle(MmToPixel(0.20)).ClosingCircle(MmToPixel(0.20));
						MinValue = minSize;
						MaxValue = maxSize;
					}
					else
					{						
				    	slider = pusher.FilterHSVForSlider(poHImg->getImg(), Roi,  5, 50, 5.0, 60.0, 55.0, 150.0,true);//5, 50, 5.0, 60.0, 55.0, 150.0,true
						//slider = SortRegionArea(slider.Connection(), true)[0]./*ErosionCircle(MmToPixel(0.10)).*/DilationCircle(MmToPixel(0.20));
						slider = SortRegionArea(slider.Connection(), true)[0].ClosingCircle(MmToPixel(0.10)).OpeningCircle(MmToPixel(0.10));
						MinValue = minSize;
						MaxValue = maxSize;
					}				
					if (Paintings == 1) CHalBase::PaintRegionFilled(slider, WindowWithPaintings, 0, 255, 0);
					Value = PixelToMm2(slider.Area());
					Logger::getInstance().LogInfo("5.2 stop", CameraNo);
					// 5.3 check slider missing
					// check slider size
					if (InspectionActive(InspectionNo + 2) && (Value < MinValue || Value > MaxValue))
					{
						snprintf(str_de, sizeof(str_de), "Sliderfläche: %.2f mm² (Min %.2f, Max %.2f)", Value, MinValue, MaxValue);
						snprintf(str_en, sizeof(str_en), "Slider area: %.2f mm² (Min %.2f, Max %.2f)", Value, MinValue, MaxValue);
						SelectLanguage(RejectMessage, str_de, str_en);
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo +2, RejectMessage, PlcRejectSliderError, Roi);
						CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth);
						break;
					}

					// set data chart
					MeasurementChartSet(MeasurementChartNo, Value, MinValue, MaxValue, MinValue - 2.0, MaxValue + 2.0);//28
				}//FOR
			


				// paint inspection name as text
				if (Paintings)
				{
					int textAlignmentX = eAlignCenter;
					int textAlignmentY = eAlignBottom;
					Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);
					Y1 = Row2 + MmToPixel(1.00);
					snprintf(str_en, sizeof(str_en), sliderName);
					SelectLanguage(string1, str_de, str_en);
					CHalBase::PaintText(string1, Col1 / 2 + Col2 / 2, Y1, WindowWithPaintings, R, G, B, FontSize, HW_FONT_ARIAL, true, false, 0, textAlignmentX, textAlignmentY, true);
				}
			
			Logger::getInstance().LogInfo("5.1 slider check stop", CameraNo);
			TimeMeasurementStop();
		}

//		////////////////////////
//		// 5.1 PE foot missing  don't use
//#pragma region
//#ifdef AP3_PeFootUsed
//		InspectionNo			= 24;
//		RejectStatisticGroupNo	= 5;
//		RejectStatisticEntryNo	= 1;
//		MeasurementChartNo		= 18;
//		if (!RejectPole() && InspectionActive(InspectionNo))
//		{
//			TimeMeasurementStart();
//
//			snprintf(roiName, sizeof(roiName), "%s", article.RoiPeFoot());
//			bool expected = ArticleParameter(3) != AP3_no_PE_Foot;
//			checkViaDeviation(oResHousing, roiName, PlcRejectPeFoot, 10, 40, 20, 70, &expected);
//
//			TimeMeasurementStop();
//		}
//#endif AP3_PeFootUsed
//#pragma endregion
//
//
//
//		////////////////////////
//		// 6.1 PV metal missing
//#pragma region
//#ifdef AP4_PVMetalUsed
//		InspectionNo			= 27;
//		RejectStatisticGroupNo	= 6;
//		RejectStatisticEntryNo	= 1;
//		MeasurementChartNo		= 19;
//		if (!RejectPole() && InspectionActive(InspectionNo))
//		{
//			TimeMeasurementStart();
//
//			bool expected = ArticleParameter(4) != AP4_no_PV_Metal;
//			size_t pvMetalNo = 0;
//			do
//			{
//				snprintf(roiNameNumbered, sizeof(roiName), "%s%i", article.RoiPvMetal(), ++pvMetalNo);
//				if (poSMs->findROI(oResHousing, roiNameNumbered) >= 0)
//					checkViaDeviation(oResHousing, roiNameNumbered, PlcRejectPvMetal, 50, 90, 60, 100, &expected);
//				else
//					break;
//			} while (!RejectPole());
//
//			TimeMeasurementStop();
//		}
//#endif AP4_PVMetalUsed
//#pragma endregion

		// Show all pusher result text
		ShowResultPusher(WindowWithPaintings, eAlignCenter, eAlignBottom, false);
		ShowResultPusher(WindowAdditional2, eAlignCenter, eAlignBottom, true);



		////////////////////////
		// 6.2/6.3 PE foot rail dimension / PE foot insert distance left / right
#pragma region
//#ifdef AP7_PeFootRailDimensionUsed
		InspectionNo = 25;
		RejectStatisticGroupNo = 6;//for showing 
		RejectStatisticEntryNo = 2;
		MeasurementChartNo = 21;
		Paintings = PaintingsEditable1();
		if ((!RejectPole() && ArticleParameter(3) != AP3_no_PE_Foot)&& (InspectionActive(25)|| InspectionActive(26)))
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("6.2 PE foot rail dimension started", CameraNo);
			double res1_x, res2_x;
			char PEModelName[200];
			HRegion roiPEDimension = HRegion::GenEmptyRegion();
			SMLibResult oResPEDimension;
			InitSMResult(&oResPEDimension);

			// search model at left side
			snprintf(PEModelName, sizeof(PEModelName), "%s", article.RoiPeRailFoot(1).c_str());
			bool expected = ArticleParameter(3) != AP3_no_PE_Foot;
			poSMs->getRoiAndParams(oResHousing, PEModelName, roiPEDimension);
			if (!roiPEDimension.IsEmpty())
			{
				
				HImagePE_Foot_Check = *poHImg->getImg();
				FilterWindowNo = WindowCamera;	// set filter to result target window

				// PE contrast filter to get more constant shape position (left & right). Test with fixed position.
				if (InspectionActive(32))
				{
					int MinLum = 55;
					int MaxLum = 120;
					MinLum = 80;
					MaxLum = 150;
					for (size_t filterNo = 0; filterNo < 2; filterNo++)
					{
						int ix1 = 350;
						int ix2 = 650;
						int iy1 = 680;
						int iy2 = 950;
						//					poSMs->getRoiAndParams(oResHousing, "ContrastFilter", Roi);
						//					Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);

						snprintf(roiNameNumbered, sizeof(roiNameNumbered), "ContrastFilter%i", filterNo + 1);
						poSMs->getRoiAndParams(oResHousing, roiNameNumbered, Roi);
						if (Roi.IsEmpty())
						{
							snprintf(str_de, sizeof(str_de), "ROI für Pusher %i fehlt!", filterNo + 1);
							snprintf(str_en, sizeof(str_en), "ROI of pusher %i is missing!", filterNo + 1);
							//						SelectLanguage(RejectMessage, str_de, str_en);
							//						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
							//						break;
						}
						else
						{
							Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);
							ix1 = Col1;
							ix2 = Col2;
							iy1 = Row1;
							iy2 = Row2;

							Filter_Helligkeit_spreizen(1, CameraNo,				// Nummer (nur Anzeige), KameraNummer
								ix1, iy1,//350, 680,				// Linke obere Ecke (Messfeld)
								ix2, iy2,//650, 950,				// Rechte untere Ecke (Messfeld)
								1,						// Prozent Schwelle
								0,						// Anzahl der Pixel für glätten (X-Richtung)
								0,						// Anzahl der Pixel für glätten (Y-Richtung)
								MinLum,					// Mindest Helligkeit (untere Grenze beim Spreizen)
								MaxLum,					// Maximale Helligkeit (obere Schwelle beim Spreizen)
								1,						// 0 Automatische Schwellwert (Prozent Schwelle ca. 0-0.05)
														// 1 Feste Grenzwerte werden übergeben
														// 2 Maximale Spreizung ohne Sättigung
														// 3 Autom. Summen-Schwellwert (Prozent Schwelle ca. 0-5)
								WindowCamera,			// Nummer des Quellbildes
								WindowAdditional1,		// Nummer des Zielbildes
								WindowAdditional1,		// Nummer des Anzeigebildes
								1);						// Einzeichnen
						}
					}
/*
					Filter_Helligkeit_spreizen(1, CameraNo,				// Nummer (nur Anzeige), KameraNummer
						1300, 680,				// Linke obere Ecke (Messfeld)
						1500, 950,				// Rechte untere Ecke (Messfeld)
						1,						// Prozent Schwelle
						0,						// Anzahl der Pixel für glätten (X-Richtung)
						0,						// Anzahl der Pixel für glätten (Y-Richtung)
						MinLum,					// Mindest Helligkeit (untere Grenze beim Spreizen)
						MaxLum,					// Maximale Helligkeit (obere Schwelle beim Spreizen)
						1,						// 0 Automatische Schwellwert (Prozent Schwelle ca. 0-0.05)
												// 1 Feste Grenzwerte werden übergeben
												// 2 Maximale Spreizung ohne Sättigung
												// 3 Autom. Summen-Schwellwert (Prozent Schwelle ca. 0-5)
						WindowCamera,			// Nummer des Quellbildes
						WindowAdditional1,		// Nummer des Zielbildes
						WindowAdditional1,		// Nummer des Anzeigebildes
						1);						// Einzeichnen
*/
					FilterWindowNo = WindowAdditional1;	// set filter to result target window
					HImagePE_Foot_Check = CHalBase::RgbhToHImage(FilterWindowNo, CameraNo, 0);
				}


				// set minimum/maximum score
				MinValue = 80;//85
				if (PECalibration_active) MinValue = 65;

				// set name of ShapeModel
				ShapeModelActive = PEModelName;

				// Calibration mode / point
				ShapeModelActivate = 0x01;
				if (Artikel.Typ[2] == AP2_QUATTRO) ShapeModelActivate = 0x02;
				if (PECalibration_active) ShapeModelActivate = 0x04;


				Hlong r1, c1, r2, c2;
				//poSMs->getRoiAndParams(oResHousing, PEModelName, roiPEDimension);
				//poSMs->getRoiAndParams(oResHousing, roiNameNumbered, Roi);
				r1 = roiPEDimension.SmallestRectangle1(&c1, &r2, &c2);
				CenterRow = roiPEDimension.SmallestRectangle2(&CenterCol, nullptr, &Length1, &Length2);

				// set ROI
				X1 = (int)CenterCol;
				Y1 = (int)CenterRow;
				Width = MmToPixel(3.00);
				Height = MmToPixel(3.00);

				// search ShapeModel
				poSMs->find(
					HImagePE_Foot_Check,		// search image
					ShapeModelActive,			// name of ShapeModel
					X1, Y1,						// search position X/Y, default=0/0 for using ShapeModel settings
					Width, Height,				// search position width/height, default=0/0 for using ShapeModel settings
//					roiPEDimension,				// search region
					ShapeModelActivate,			// activate ShapeModel (0 for using ShapeModel settings)
					true);						// true:	ShapeModel should be found (MinValue used)
												// false:	ShapeModel should not be found (MaxValue used)

				// get ShapeModel results
				oResPEDimension = poSMs->res(ShapeModelActive,	// name of ShapeModel
					MeasurementChartNo,			// measurement chart number for score, default=-1
					MinValue,					// minimum / maximum score
					Paintings,					// activate paintings, default=1
					LineWidth,					// width paintings, default=1
					0);							// index, default=0
			    res1_x = oResPEDimension.x_ref_point;
				// ShapeModel was not found
				if (!poSMs->Err(ShapeModelActive))
				{
					// Messpunkte eintragen
					PointNo = 1;
					if (!PECalibration_active)
					{
						Set_Measurement_Point(CameraNo, PointNo, oResPEDimension.x_ref_point, oResPEDimension.y_ref_point, "LP=Ref", 3, -1);
					}
					else
					{
						double TempX = 0.00;
						double TempY = 0.00;
						Set_Measurement_Point(CameraNo, PointNo, oResPEDimension.x_ref_point, oResPEDimension.y_ref_point + MmToPixelDouble(PECalibration_YOffset), "LP=Ref", 3, -1);
						ProjectionPoint_Calculate(oResPEDimension.x_ref_point, oResPEDimension.y_ref_point,	// Input Origin point [pixel]
							oResPEDimension.angle_deg,												// Rotation angle [° in GRAD], 0° means no rotation, psotiv angle counter clockwise
							MmToPixelDouble(0.00), MmToPixelDouble(PECalibration_YOffset),			// [Pixel], positive values to the right / bottom
							&TempX, &TempY);														// Return values
						Set_Measurement_Point(CameraNo, PointNo, TempX, TempY, "LP=Ref", 3, -1);
					}
				}
				else // ShapeModel was not found
				{
					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, poSMs->getErr(ShapeModelActive), PlcRejectPeRailDimension, poSMs->getSearchRegion(ShapeModelActive));
				}

				// Enhanced shape model result paintings (scoreResult)
				if (Paintings > 0)
					EnhancedModelResultPaintings(oResPEDimension, -MmToPixel(2.50), +MmToPixel(2.50), /*X1, Y1,*/ true, MinValue, WindowWithPaintings, 100, R, G, B);
			}
			else
			{
				// Error ROI nicht definiert
				snprintf(str_de, sizeof(str_de), "ROI %s fehlt!", PEModelName);
				snprintf(str_en, sizeof(str_en), "ROI %s is missing!", PEModelName);
				SelectLanguage(RejectMessage, str_de, str_en);
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
			}

			// search model at right side (only if left side was found)
			snprintf(PEModelName, sizeof(PEModelName), "%s", article.RoiPeRailFoot(2).c_str());
			expected = ArticleParameter(3) != AP3_no_PE_Foot;
			poSMs->getRoiAndParams(oResHousing, PEModelName, roiPEDimension);
			if (!RejectPole())
			{
				if (!roiPEDimension.IsEmpty())
				{
					// set minimum/maximum score
					MinValue = 85;

					// set name of ShapeModel
					ShapeModelActive = PEModelName;
					ShapeModelActivate = 0x00;

					// search ShapeModel
					poSMs->find(HImagePE_Foot_Check,ShapeModelActive,	// name of ShapeModel
						roiPEDimension,				// search region
						ShapeModelActivate,			// activate ShapeModel (0 for using ShapeModel settings)
						true);						// true:	ShapeModel should be found (MinValue used)
													// false:	ShapeModel should not be found (MaxValue used)

					// get ShapeModel results
					oResPEDimension = poSMs->res(ShapeModelActive,	// name of ShapeModel
						MeasurementChartNo + 1,		// measurement chart number for score, default=-1
						MinValue,					// minimum / maximum score
						Paintings,					// activate paintings, default=1
						LineWidth,					// width paintings, default=1
						0);							// index, default=0
					res2_x = oResPEDimension.x_ref_point;
					double resx = abs(res2_x - res1_x);
					// ShapeModel was not found
					if (!poSMs->Err(ShapeModelActive))
					{
						// Messpunkte eintragen
						PointNo = 2;
						Set_Measurement_Point(CameraNo, PointNo, oResPEDimension.x_ref_point, oResPEDimension.y_ref_point, "RP=Ref", 4, -1);
					}
					else // ShapeModel was not found
					{
						// set reject
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, poSMs->getErr(ShapeModelActive), PlcRejectPeRailDimension, poSMs->getSearchRegion(ShapeModelActive));
					}

					// Enhanced shape model result paintings (scoreResult)
					if (Paintings > 0)
					{
						double dX1, dY1; roiPEDimension.AreaCenter(&dY1, &dX1); X1 = (int)dX1; Y1 = (int)dY1;
						EnhancedModelResultPaintings(oResPEDimension, +MmToPixel(3.00), +MmToPixel(1.00),/* X1, Y1,*/ true, MinValue, WindowWithPaintings, 100, R, G, B);
					}
				}
				else
				{
					// Error ROI nicht definiert
					snprintf(str_de, sizeof(str_de), "ROI %s fehlt!", PEModelName);
					snprintf(str_en, sizeof(str_en), "ROI %s is missing!", PEModelName);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
				}
			}
		
			////////////////////////
			// Objects found -> Calculate / check rail dimension
			//6.2 6.3 C6-I13 Check PE rail dimension
			if (!RejectPole() && (InspectionActive(InspectionNo) || InspectionActive(InspectionNo + 1)))
			{
				MinValue = MIN_RailDimension;	// mm
				MaxValue = MAX_RailDimension;	// mm

				  // set tolerance
				  // see document 00867822
				Error = false;
				switch (ArticleParameter(3))
				{
				case AP3_PE_Foot_QT_2_5:		// 0055133
				case AP3_PE_Foot_STTB_2_5:		// 0055134
				case AP3_PE_Foot_STS_2_5:		// 0055377
				case AP3_PE_Foot_PTS_4:			// 9732654	
					MinValue = 34.32 - 0.20;
					MaxValue = 34.32 + 0.20;
					break;

				case AP3_PE_Foot_QTC_2_5:		// 0055135
				case AP3_PE_Foot_ST_2_5_2P:		// 0055138
				case AP3_PE_Foot_DTI_2_5:		// 0053512
				case AP3_PE_Foot_PITS_2_5:		// 0110480
					MinValue = 34.40 - 0.15;
					MaxValue = 34.40 + 0.15;
					break;

				case AP3_PE_Foot_ST_16:			// 0084238
				case AP3_PE_Foot_ST_35:			// 0084237
					MinValue = 34.40 - 0.15;
					MaxValue = 34.40 + 0.15;
					break;

				case AP3_PE_Foot_PIT_1_5:		// 0118840
				case AP3_PE_Foot_PITS_1_5:		// 0123030
				case AP3_PE_Foot_UTI_6:			// 1032440
					MinValue = 34.50 - 0.15;
					MaxValue = 34.50 + 0.15;
					break;

				case AP3_PE_Foot_MSB_2_5:		// 0094414
				case AP3_PE_Foot_MPT_1_5:		// 0145777
					MinValue = 14.50 - 0.15;
					MaxValue = 14.50 + 0.15;
					break;

				case AP3_PE_Foot_MSB_NS35_2_5:	// 0097782
					MinValue = 34.40 - 0.15;
					MaxValue = 34.40 + 0.15;
					break;

				case AP3_PE_Foot_UT_2_5:		// 0032999
				case AP3_PE_Foot_UTTB_2_5:		// 0055511
					MinValue = 31.94 - 0.15;
					MaxValue = 31.94 + 0.15;
					break;

				case AP3_PE_Foot_PT_1_5:		// 0152815 for PV machine
					/*MinValue = 32.20 - 0.15;
					MaxValue = 32.20 + 0.15;*/
					MinValue = 32.40 - 0.15;
					MaxValue = 32.40 + 0.15;
					// Calibration mode / point
					if (PECalibration_active)
					{
						MinValue = 32.40 - 0.15;
						MaxValue = 32.40 + 0.15;
					}
					break;

				case AP3_PE_Foot_UT_16:			// 0175135
				case AP3_PE_Foot_UT_35:			// 0175136
					MinValue = 32.16 - 0.15;
					MaxValue = 32.16 + 0.15;
					break;

				case AP3_PE_Foot_MUT_2_5:		// 0139055
					MinValue = 14.5 - 0.15;
					MaxValue = 14.5 + 0.15;
					break;

				default:
					// set reject message
					snprintf(str_de, sizeof(str_de), "Parameter für PE Fuss Typ falsch gesetzt.");
					snprintf(str_en, sizeof(str_en), "Parameter for PE foot typ not correct.");
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPeRailDimension);
					break;
				}



				// Winkel der Ausgleichsgeraden setzen (normalerweise Gehäusewinkel)
				// Set base angle (normally housing angle)
				double CorrectionAngle = AngleDeg;

				// Calculation of offset point on the left: End point of the projection line with larger Y-distance
				int MesspunkteAbstand_Y = (int)Differenz(MeasurementPoint[CameraNo][1].Result_Point_Y, MeasurementPoint[CameraNo][2].Result_Point_Y);
				int ProjektionAbstand_Y = 40;
				double pX_L = Kreispunkt_X_double(MeasurementPoint[CameraNo][1].Result_Point_X, CorrectionAngle, 2.00 * (MesspunkteAbstand_Y + ProjektionAbstand_Y));
				double pY_L = Kreispunkt_Y_double(MeasurementPoint[CameraNo][1].Result_Point_Y, CorrectionAngle, 2.00 * (MesspunkteAbstand_Y + ProjektionAbstand_Y));

				// Enter end point of projection line for display
				PointNo = 3;
				Set_Measurement_Point(CameraNo, PointNo, pX_L, pY_L, "LP + Rotation", 0, 4);

				// Calculation of offset point on the right: with defined Y-distance (for projection on projection lines on the left)
				double pX_R = Kreispunkt_X_double(MeasurementPoint[CameraNo][2].Result_Point_X, CorrectionAngle, ProjektionAbstand_Y);
				double pY_R = Kreispunkt_Y_double(MeasurementPoint[CameraNo][2].Result_Point_Y, CorrectionAngle, ProjektionAbstand_Y);

				// Calculate projection point on line left
				double pX_Porj_L = 0.00;
				double pY_Porj_L = 0.00;
				projection_pl(pY_R, pX_R,										// Calculated (offset) points right
							  MeasurementPoint[CameraNo][1].Result_Point_Y,		// Start point (Y) projection line left
							  MeasurementPoint[CameraNo][1].Result_Point_X,		// Start point (X) projection line left
							  pY_L, pX_L,										// End point projection line left
							  &pY_Porj_L, &pX_Porj_L);							// Result: Point on projection line left

				PointNo = 4;
				//Set_Measurement_Point(CameraNo, PointNo, MeasurementPoint[CameraNo][2].Result_Point_X, MeasurementPoint[CameraNo][2].Result_Point_Y + 50, "RightPoint", 0, 0);
				Set_Measurement_Point(CameraNo, PointNo, pX_R, pY_R, "RP + Rotation", PointNo + 1, 0);						// Punkt rechts
				MeasurementPoint[CameraNo][PointNo].SetMinValue = MinValue;					// Vorgabe MinWert	(für farbliche Markierung DistanceText)
				MeasurementPoint[CameraNo][PointNo].SetMaxValue = MaxValue;					// Vorgabe MinWert	(für farbliche Markierung DistanceText)

				// Enter measuring points
				PointNo = 5;
				//Set_Measurement_Point(CameraNo, PointNo, MeasurementPoint[CameraNo][1].Result_Point_X, MeasurementPoint[CameraNo][2].Result_Point_Y + 50, "LeftPoint", PointNo + 1, 0);
				Set_Measurement_Point(CameraNo, PointNo, pX_Porj_L, pY_Porj_L, "LeftProjetionPoint", 0, 0);	// Result of projection on left side
				//MeasurementPoint[CameraNo][PointNo].SetMinValue = MinValue;					// Vorgabe MinWert	(für farbliche Markierung DistanceText)
				//MeasurementPoint[CameraNo][PointNo].SetMaxValue = MaxValue;					// Vorgabe MinWert	(für farbliche Markierung DistanceText)

				// Calculate distance in space (including angle of the regression line)
				Value = Laenge_berechnen_MM(CameraNo, 
					MeasurementPoint[CameraNo][4].Result_Point_X, MeasurementPoint[CameraNo][4].Result_Point_Y,
					MeasurementPoint[CameraNo][5].Result_Point_X, MeasurementPoint[CameraNo][5].Result_Point_Y);
				Value += 0.06;
					

				
				// Error position
				X1 = (int)MeasurementPoint[CameraNo][1].Result_Point_X;
				Y1 = (int)MeasurementPoint[CameraNo][1].Result_Point_Y;
				X2 = (int)MeasurementPoint[CameraNo][2].Result_Point_X;
				Y2 = (int)MeasurementPoint[CameraNo][2].Result_Point_Y;

				// check result
				if (InspectionActive(InspectionNo) && (Value < MinValue))
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Abstand zu klein = %.3f (Min %.2f)", Value, MinValue);
					snprintf(str_en, sizeof(str_en), "Distance 32.4 too low = %.3f (Min %.2f)", Value, MinValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPeRailDimension, X1, Y1, X2, Y2);
				}

				// check result
				if (InspectionActive(InspectionNo + 1) && (Value > MaxValue))
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Abstand zu gross = %.3f (Max %.2f)", Value, MaxValue);
					snprintf(str_en, sizeof(str_en), "Distance 32.4 too high = %.3f (Max %.2f)", Value, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectPeRailDimension, X1, Y1, X2, Y2);
				}

				// set data chart
				MeasurementChartSet(MeasurementChartNo+2, Value, MinValue, MaxValue);//23


				// Display PE-RailDimension
				snprintf(str_de, sizeof(str_de), "PE-Abstand");
				snprintf(str_en, sizeof(str_en), "RailDimension");
				SelectLanguage(string1, str_de, str_en);
				Display_ResultValueBar(CameraNo,	// Kamera Nummer
					WindowWithPaintings,			// Bild, in dass das Mass eingetragen wird
					// Position
					Mittelwert(MeasurementPoint[CameraNo][4].Result_Point_X, MeasurementPoint[CameraNo][5].Result_Point_X),		// X-Position Bezugspunkt 
					Mittelwert(MeasurementPoint[CameraNo][4].Result_Point_Y, MeasurementPoint[CameraNo][5].Result_Point_Y),		// Y-Position Bezugspunkt 
					MmToPixel(0.00),				// X-distance to reference point (used position if ! OffsetFromRef)
					MmToPixel(1.85),				// Y-distance to reference point (position used if ! OffsetFromRef)
					true,							// Incorrect if the offset positions are to be used as absolute positions without the reference point
					// Design
					1,								// Art der Anzeige (0: RT/GT/BT für Text & Balken, 1: Farbe nach Ergebnis berechnen)
					(int)(FontSize * 7.0),			// Breite des Balkens (-1: Standard, >=1: Breite vorgeben)
					FontSize,						// Schriftgröße (Zeichensatz)
					255, 255, 255,					// Farbe des Textes (R/G/B)
					// Values / Borders
					Value,							// ScoreResult vom Model 0-100%
					"mm",							// Text für Einheit (Standard: %)
					MinValue,						// Minimum erlaubter Wert (UTG)
					MaxValue,						// Minimum erlaubter Wert (UTG)
					Mittelwert(MinValue, MaxValue),	// Normaler Wert (bei +/- Verteilung ist der Standard 0)
					MinValue - 0.00,				// Anzeige MinWert 0-100%
					MaxValue + 0.00,				// Anzeige MaxWert 0-100%
					// zusätzliche Texte
					string1,						// Name der Modell-Suche
					"");							// Name der Kontrolle

				// set data chart PE foot dimension in pixel for calibration
			
			}	// End of if (!RejectPole())


			////////////////////////
			// x.x Housing edge for PE insert distance left / right

			double RegressionAngle = AngleDeg;
			int BasePoint = 6;					// for projecion line Setztiefe links/rechts
			InspectionNo = 27;					// 1. Inspection that needs housing position

			if (!Reject() && 
				(InspectionActive(InspectionNo)		|| 
				 InspectionActive(InspectionNo + 1) ||
				 InspectionActive(InspectionNo + 2) ||
				 InspectionActive(InspectionNo + 3)))
			{
				RoiName = "PeFootHousingPosition"; // fill in region name in model
				poSMs->getRoiAndParams(oResHousing, RoiName, Roi);
				if (Roi.IsEmpty())
				{
					snprintf(str_de, sizeof(str_de), "ROI '%s' nicht definiert!", RoiName);
					snprintf(str_en, sizeof(str_en), "ROI '%s' isn't defined!", RoiName);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, 0, 0, ImageWidth(), ImageHeight());
				}
				else
				{
					// Set / caculate housing position
					double Length1 = 0;
					double Length2 = 0;
					double CenterRow = 0;
					double CenterCol = 0;

					// Step 1: Use ROI position (no double position values)
					Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);
					PointNo = 6;
					Set_Measurement_Point(CameraNo, PointNo, Mittelwert(Col1, Col2), Mittelwert(Row1, Row2), "Housing Pos.", 7, -1);	// Punkt am Gehäuse

					// Step 2: Use ROI position incl. rotation (double values, if witdh of region is very big) 
					CenterRow = Roi.SmallestRectangle2(&CenterCol, nullptr, &Length1, &Length2);
					PointNo = 6;
					Set_Measurement_Point(CameraNo, PointNo, CenterCol, CenterRow, "Housing Pos.", PointNo + 2, -1);						// Punkt am Gehäuse

					// Step 3: Use ROI area to get center of area (double center values a little bit better than step 3)
					Roi.AreaCenter(&CenterRow, &CenterCol);
					PointNo = 6;
					Set_Measurement_Point(CameraNo, PointNo, CenterCol, CenterRow, "Housing Pos.", PointNo + 2, -1);						// Punkt am Gehäuse

					// Step 4: detect position and rotation of bottom housing edges
					if (InspectionActive(31))	// detect housing edge instead of found region
					{
						double Bottom_edge_left = 0.00;
						double Bottom_edge_right = 0.00;
						int x1 = (int)(CenterCol + Length1) - 100;
						int x2 = (int)(CenterCol + Length1) + 0;
						Bottom_edge_right = Subpixel_Uebergang_v2(1,					// Verlaufsnummer für Array + Anzeige
							CameraNo,					// Nummer der Kamera
							"HousingRight.",			// Bezeichnung des Verlaufs
							x1, (int)CenterRow - 15,	// Linke obere Ecke des Feldes
							x2, (int)CenterRow + 25,	// Rechte untere Ecke des Feldes
							10,							// Richtung:  0 = Zeilenweise oben nach unten(from top->bottom)
														// Richtung: 10 = Zeilenweise unten nach oben (from bottom->top)
														// Richtung:  1 = Spaltenweise links nach rechts(from left->right)
														// Richtung: 11 = Spaltenweise rechts nach links(from right->left)
							Sub_DunkelHell,				// Sub_HellDunkel, Sub_DunkelHell oder Sub_Auto from dark->bright
							100,						// x-faches Subpixling (5 - 1000)
							WindowCamera, WindowWithPaintings,					// Original-/ Anzeigefensternummer
							9);							// Anzeige der Feldgrenzen

						PointNo = 6;
						Set_Measurement_Point(CameraNo, PointNo, Mittelwert(x1, x2), Bottom_edge_right, "Housing Edge", PointNo + 2, -1);						// Punkt am Gehäuse

						x1 = (int)(CenterCol - Length1);
						x2 = (int)(CenterCol - Length1) + 100;
						Bottom_edge_left = Subpixel_Uebergang_v2(2,					// Verlaufsnummer für Array + Anzeige
							CameraNo,					// Nummer der Kamera
							"HousingLeft.",			// Bezeichnung des Verlaufs
							x1, (int)CenterRow - 35,	// Linke obere Ecke des Feldes
							x2, (int)CenterRow + 25,	// Rechte untere Ecke des Feldes
							10,							// Richtung:  0 = Zeilenweise oben nach unten
														// Richtung: 10 = Zeilenweise unten nach oben
														// Richtung:  1 = Spaltenweise links nach rechts
														// Richtung: 11 = Spaltenweise rechts nach links
							Sub_DunkelHell,				// Sub_HellDunkel, Sub_DunkelHell oder Sub_Auto
							100,						// x-faches Subpixling (5 - 1000)
							WindowCamera, WindowWithPaintings,					// Original-/ Anzeigefensternummer
							9);							// Anzeige der Feldgrenzen

						PointNo = 7;
						Set_Measurement_Point(CameraNo, PointNo, Mittelwert(x1, x2), Bottom_edge_left, "Housing Edge", PointNo + 1, -1);// Point on the housing
						//水平轴（正X轴）逆时针旋转到线段方向的夹角
						RegressionAngle = (-1.0) * (90.0 - Winkel_berechnen(MeasurementPoint[CameraNo][7].Result_Point_X, MeasurementPoint[CameraNo][7].Result_Point_Y,
							MeasurementPoint[CameraNo][6].Result_Point_X, MeasurementPoint[CameraNo][6].Result_Point_Y));

						MeasurementPoint[CameraNo][BasePoint].RelationToPoint = 0;
						BasePoint = 7;
					}
				}
			}


			////////////////////////
			// 6.4 PE insert distance left (0.98)-->0.17
			// C6 - I11 Check PE left dimension
			InspectionNo = 27;
			RejectStatisticGroupNo = 6;
			RejectStatisticEntryNo = 4;
			MeasurementChartNo = 25;

			if (!Reject() && (InspectionActive(InspectionNo) || InspectionActive(InspectionNo + 1)))
			{
				// Measurement point [6] = Bottom edge right
				// Measurement point [7] = Bottom edge left

				//MinValue = 0.98 - 0.105;	// Insertion depth left Min.
				MaxValue = 0.17 + 0.20;	// Insertion depth left Max.

				// 2024-03-14: Temporary for production until clarification of dimensions / tolerance
				MinValue = 0.17 - 0.15-0.03;	// Einstecktiefe links Min.
				//if (Artikel.Typ[2] != AP2_QUATTRO) MinValue = 0.80;	// Einstecktiefe links Min.

				// Calculation of offset point on the left: End point of the projection line with larger X-distance
				int MesspunkteAbstand_X = (int)Differenz(MeasurementPoint[CameraNo][1].Result_Point_X, MeasurementPoint[CameraNo][BasePoint].Result_Point_X);
				int ProjektionAbstand_X = 90;
				double CorrectionAngle = RegressionAngle - 90;
				double pX_L = Kreispunkt_X_double(MeasurementPoint[CameraNo][BasePoint].Result_Point_X, CorrectionAngle, 1.25 * (MesspunkteAbstand_X + ProjektionAbstand_X));
				double pY_L = Kreispunkt_Y_double(MeasurementPoint[CameraNo][BasePoint].Result_Point_Y, CorrectionAngle, 1.25 * (MesspunkteAbstand_X + ProjektionAbstand_X));

				// Endpunkt der Projektionslinie für Anzeige eintragen
				PointNo = 8;
				Set_Measurement_Point(CameraNo, PointNo, pX_L, pY_L, "Housing + Rotation", 0, 1);

				// Berechnung versetzter Punkt rechts: mit definiertem Y-Abstand (für Projektion auf Projektionsliniene links)
				double pX_R = Kreispunkt_X_double(MeasurementPoint[CameraNo][1].Result_Point_X, CorrectionAngle, ProjektionAbstand_X);
				double pY_R = Kreispunkt_Y_double(MeasurementPoint[CameraNo][1].Result_Point_Y, CorrectionAngle, ProjektionAbstand_X);

				// Projektionspunkt auf Linie von Gehäuseunterkante
				double pX_Porj_L = 0.00;
				double pY_Porj_L = 0.00;
				projection_pl(pY_R, pX_R,								// Berechneter (versetzter) Punkte rechts
					MeasurementPoint[CameraNo][BasePoint].Result_Point_Y,		// Startpunkt (Y) Pojektionslinie links
					MeasurementPoint[CameraNo][BasePoint].Result_Point_X,		// Startpunkt (X) Pojektionslinie links
					pY_L, pX_L,											// Endpunkt Pojektionslinie links
					&pY_Porj_L, &pX_Porj_L);							// Ergebnis: Punkt auf Projektionslinie links

				PointNo = 9;
				//Set_Measurement_Point(CameraNo, PointNo, MeasurementPoint[CameraNo][2].Result_Point_X, MeasurementPoint[CameraNo][2].Result_Point_Y + 50, "RightPoint", 0, 0);
				Set_Measurement_Point(CameraNo, PointNo, pX_R, pY_R, "RP + Rotation", PointNo + 1, 0);						// Punkt rechts
				MeasurementPoint[CameraNo][PointNo].SetMinValue = MinValue;					// Vorgabe MinWert	(für farbliche Markierung DistanceText)
				MeasurementPoint[CameraNo][PointNo].SetMaxValue = MaxValue;					// Vorgabe MinWert	(für farbliche Markierung DistanceText

				// Enter measuring points
				PointNo = 10;
				//Set_Measurement_Point(CameraNo, PointNo, MeasurementPoint[CameraNo][1].Result_Point_X, MeasurementPoint[CameraNo][2].Result_Point_Y + 50, "LeftPoint", PointNo + 1, 0);
				Set_Measurement_Point(CameraNo, PointNo, pX_Porj_L, pY_Porj_L, "HousingProjetionPoint", 0, 0);	// Ergebnis der Projektion auf linke Seite
				//MeasurementPoint[CameraNo][PointNo].SetMinValue = MinValue;					// Vorgabe MinWert	(für farbliche Markierung DistanceText)
				//MeasurementPoint[CameraNo][PointNo].SetMaxValue = MaxValue;					// Vorgabe MinWert	(für farbliche Markierung DistanceText

				CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 250, 250, 1);


				// Abstand im Raum (incl. Winkel der Ausgleichsgeraden) berechnen
				Value = Laenge_berechnen_MM(CameraNo,
					MeasurementPoint[CameraNo][9].Result_Point_X, MeasurementPoint[CameraNo][9].Result_Point_Y,
					MeasurementPoint[CameraNo][10].Result_Point_X, MeasurementPoint[CameraNo][10].Result_Point_Y);
				Value -= 0.01;

				// Fehlerposition
				X1 = (int)MeasurementPoint[CameraNo][9].Result_Point_X;
				Y1 = (int)MeasurementPoint[CameraNo][9].Result_Point_Y;
				X2 = (int)MeasurementPoint[CameraNo][10].Result_Point_X;
				Y2 = (int)MeasurementPoint[CameraNo][10].Result_Point_Y;

				// check result
				if (InspectionActive(InspectionNo) && (Value < MinValue))
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Abstand zu klein = %.3f (Min %.3f)", Value, MinValue);
					snprintf(str_en, sizeof(str_en), "Distance 0.17 too low = %.3f (Min %.3f)", Value, MinValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPeRailDimension, X1, Y1, X2, Y2);
				}

				// check result
				if (InspectionActive(InspectionNo + 1) && (Value > MaxValue))
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Abstand zu gross = %.3f (Max %.3f)", Value, MaxValue);
					snprintf(str_en, sizeof(str_en), "Distance 0.17 too high = %.3f (Max %.3f)", Value, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectPeRailDimension, X1, Y1, X2, Y2);
				}

				// set data chart
				MeasurementChartSet(MeasurementChartNo, Value, MinValue, MaxValue);//25

				// Display PE-Foot distance left side
				snprintf(str_de, sizeof(str_de), "PE-Tiefe Li.");
				snprintf(str_en, sizeof(str_en), "Deep Left");
				SelectLanguage(string1, str_de, str_en);
				Display_ResultValueBar(CameraNo,	// Kamera Nummer
					WindowWithPaintings,			// Bild, in dass das Mass eingetragen wird
					// Position
					MeasurementPoint[CameraNo][1].Result_Point_X,		// X-Position Bezugspunkt 
					MeasurementPoint[CameraNo][1].Result_Point_Y,		// Y-Position Bezugspunkt 
					MmToPixel(4.00),				// X-Abstand zum Bezugspunkt (genutzte Position, wenn !VersatzVonRef)
					MmToPixel(4.65),				// Y-Abstand zum Bezugspunkt (genutzte Position, wenn !VersatzVonRef)
					true,							// Falsch wenn die Versatzpositionen als Absolutpositionen ohne den Bezugspunkt verwendet werden sollen
					// Design
					1,								// Art der Anzeige (0: RT/GT/BT für Text & Balken, 1: Farbe nach Ergebnis berechnen)
					(int)(FontSize * 7.0),			// Breite des Balkens (-1: Standard, >=1: Breite vorgeben)
					FontSize,						// Schriftgröße (Zeichensatz)
					255, 255, 255,					// Farbe des Textes (R/G/B)
					// Values / Borders
					Value,							// ScoreResult vom Model 0-100%
					"mm",							// Text für Einheit (Standard: %)
					MinValue,						// Minimum erlaubter Wert (UTG)
					MaxValue,						// Minimum erlaubter Wert (UTG)
					0.98,//Mittelwert(MinValue, MaxValue),	// Normaler Wert (bei +/- Verteilung ist der Standard 0)
					MinValue - 0.00,				// Anzeige MinWert 0-100%
					MaxValue + 0.00,				// Anzeige MaxWert 0-100%
					// zusätzliche Texte
					string1,						// Name der Modell-Suche
					"");							// Name der Kontrolle
			}

			
			////////////////////////
			// 6.6 PE insert distance right (3.25)
			// C6 - I12 Check PE right dimension
			InspectionNo = 29;
			RejectStatisticGroupNo = 6;
			RejectStatisticEntryNo = 6;
			MeasurementChartNo = 26;

			if (!Reject() && (InspectionActive(InspectionNo) || InspectionActive(InspectionNo + 1)))
			{


				MinValue = 3.25 - 0.25;	// Einstecktiefe links Min.
				MaxValue = 3.25 + 0.25;	// Einstecktiefe links Max.
				affine_trans_point_2d(HomMat2DCompose, MeasurementPoint[CameraNo][2].Result_Point_Y, MeasurementPoint[CameraNo][2].Result_Point_X, &regressTransY_Point2, &regressTransX_Point2);
				affine_trans_point_2d(HomMat2DCompose, MeasurementPoint[CameraNo][6].Result_Point_Y, MeasurementPoint[CameraNo][6].Result_Point_X, &regressTransY_Point6, &regressTransX_Point6);
				PointNo = 9;
				Set_Measurement_Point(CameraNo, PointNo, regressTransX_Point2[0].D(), regressTransY_Point2[0].D(), "P2 + Rotation", 0, 0);
				PointNo = 10;
				Set_Measurement_Point(CameraNo, PointNo, regressTransX_Point6[0].D(), regressTransY_Point6[0].D(), "P6 + Rotation", 0, 0);
				BasePoint = 10;
				
				// Berechnung versezter Punkt rechts: Endpunkt der Projektionslinie mit größerem X-Abstand
				int MesspunkteAbstand_X = (int)Differenz(MeasurementPoint[CameraNo][9].Result_Point_X, MeasurementPoint[CameraNo][BasePoint].Result_Point_X);
				int ProjektionAbstand_X = 90;
				double CorrectionAngle = RegressionAngle + 90;
				double pX_L = Kreispunkt_X_double(MeasurementPoint[CameraNo][BasePoint].Result_Point_X, CorrectionAngle, 1.25 * (MesspunkteAbstand_X + ProjektionAbstand_X));
				double pY_L = Kreispunkt_Y_double(MeasurementPoint[CameraNo][BasePoint].Result_Point_Y, CorrectionAngle, 1.25 * (MesspunkteAbstand_X + ProjektionAbstand_X));

				// Endpunkt der Projektionslinie für Anzeige eintragen
				PointNo = 11;
				Set_Measurement_Point(CameraNo, PointNo, pX_L, pY_L, "Housing + Rotation", BasePoint, -1);

				// Berechnung versetzter Punkt rechts: mit definiertem Y-Abstand (für Projektion auf Projektionsliniene links)
				double pX_R = Kreispunkt_X_double(MeasurementPoint[CameraNo][9].Result_Point_X, CorrectionAngle, ProjektionAbstand_X);
				double pY_R = Kreispunkt_Y_double(MeasurementPoint[CameraNo][9].Result_Point_Y, CorrectionAngle, ProjektionAbstand_X);

				// Projektionspunkt auf Linie von Gehäuseunterkante
				double pX_Porj_L = 0.00;
				double pY_Porj_L = 0.00;
				projection_pl(pY_R, pX_R,								// Berechneter (versetzter) Punkte rechts
					MeasurementPoint[CameraNo][BasePoint].Result_Point_Y,		// Startpunkt (Y) Pojektionslinie links
					MeasurementPoint[CameraNo][BasePoint].Result_Point_X,		// Startpunkt (X) Pojektionslinie links
					pY_L, pX_L,											// Endpunkt Pojektionslinie links
					&pY_Porj_L, &pX_Porj_L);							// Ergebnis: Punkt auf Projektionslinie links

				PointNo = 12;
				//Set_Measurement_Point(CameraNo, PointNo, MeasurementPoint[CameraNo][2].Result_Point_X, MeasurementPoint[CameraNo][2].Result_Point_Y + 50, "RightPoint", 0, 0);
				Set_Measurement_Point(CameraNo, PointNo, pX_R, pY_R, "RP + Rotation", PointNo + 1, 0);						// Punkt rechts
				MeasurementPoint[CameraNo][PointNo].SetMinValue = MinValue;					// Vorgabe MinWert	(für farbliche Markierung DistanceText)
				MeasurementPoint[CameraNo][PointNo].SetMaxValue = MaxValue;					// Vorgabe MaxWert	(für farbliche Markierung DistanceText

				// Messpunkte eintragen
				PointNo = 13;
				//Set_Measurement_Point(CameraNo, PointNo, MeasurementPoint[CameraNo][1].Result_Point_X, MeasurementPoint[CameraNo][2].Result_Point_Y + 50, "LeftPoint", PointNo + 1, 0);
				Set_Measurement_Point(CameraNo, PointNo, pX_Porj_L, pY_Porj_L, "HousingProjetionPoint", 0, 0);	// Ergebnis der Projektion auf linke Seite
				//MeasurementPoint[CameraNo][PointNo].SetMinValue = MinValue;					// Vorgabe MinWert	(für farbliche Markierung DistanceText)
				//MeasurementPoint[CameraNo][PointNo].SetMaxValue = MaxValue;					// Vorgabe MaxWert	(für farbliche Markierung DistanceText

				CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 250, 250, 1);


				// Abstand im Raum (incl. Winkel der Ausgleichsgeraden) berechnen
				Value = Laenge_berechnen_MM(CameraNo,
					MeasurementPoint[CameraNo][12].Result_Point_X, MeasurementPoint[CameraNo][12].Result_Point_Y,
					MeasurementPoint[CameraNo][13].Result_Point_X, MeasurementPoint[CameraNo][13].Result_Point_Y);
				Value -= calibrationData_3_25;
				// Fehlerposition
				X1 = (int)MeasurementPoint[CameraNo][12].Result_Point_X;
				Y1 = (int)MeasurementPoint[CameraNo][12].Result_Point_Y;
				X2 = (int)MeasurementPoint[CameraNo][13].Result_Point_X;
				Y2 = (int)MeasurementPoint[CameraNo][13].Result_Point_Y;

				// check result
				if (InspectionActive(InspectionNo) && (Value < MinValue))
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Abstand zu klein = %.3f (Min %.2f)", Value, MinValue);
					snprintf(str_en, sizeof(str_en), "Distance 3.25 too low = %.3f (Min %.2f)", Value, MinValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPeRailDimension, X1, Y1, X2, Y2);
				}

				// check result
				if (InspectionActive(InspectionNo + 1) && (Value > MaxValue))
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Abstand zu gross = %.3f (Max %.2f)", Value, MaxValue);
					snprintf(str_en, sizeof(str_en), "Distance 3.25 too high = %.3f (Max %.2f)", Value, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo + 1, RejectMessage, PlcRejectPeRailDimension, X1, Y1, X2, Y2);
				}

				// set data chart
				MeasurementChartSet(MeasurementChartNo, Value, MinValue, MaxValue);//26

				// Display PE-Foot distance right side
				snprintf(str_de, sizeof(str_de), "PE-Tiefe Ri.");
				snprintf(str_en, sizeof(str_en), "Deep Right");
				SelectLanguage(string1, str_de, str_en);
				Display_ResultValueBar(CameraNo,	// Kamera Nummer
					WindowWithPaintings,			// Bild, in dass das Mass eingetragen wird
					// Position
					MeasurementPoint[CameraNo][2].Result_Point_X,				// X-Position Bezugspunkt 
					MeasurementPoint[CameraNo][2].Result_Point_Y,				// Y-Position Bezugspunkt 
					-MmToPixel(4.00),								// X-Abstand zum Bezugspunkt (genutzte Position, wenn !VersatzVonRef)
					MmToPixel(2.65),							// Y-Abstand zum Bezugspunkt (genutzte Position, wenn !VersatzVonRef)
					true,							// Falsch wenn die Versatzpositionen als Absolutpositionen ohne den Bezugspunkt verwendet werden sollen
					// Design
					1,								// Art der Anzeige (0: RT/GT/BT für Text & Balken, 1: Farbe nach Ergebnis berechnen)
					(int)(FontSize * 7.0),			// Breite des Balkens (-1: Standard, >=1: Breite vorgeben)
					FontSize,						// Schriftgröße (Zeichensatz)
					255, 255, 255,					// Farbe des Textes (R/G/B)
					// Values / Borders
					Value,							// ScoreResult vom Model 0-100%
					"mm",							// Text für Einheit (Standard: %)
					MinValue,						// Minimum erlaubter Wert (UTG)
					MaxValue,						// Minimum erlaubter Wert (UTG)
					3.25,//Mittelwert(MinValue, MaxValue),	// Normaler Wert (bei +/- Verteilung ist der Standard 0)
					MinValue - 0.00,				// Anzeige MinWert 0-100%
					MaxValue + 0.00,				// Anzeige MaxWert 0-100%
					// zusätzliche Texte
					string1,						// Name der Modell-Suche
					"");							// Name der Kontrolle
			}

			Logger::getInstance().LogInfo("6.2 PE foot rail dimension stop", CameraNo);
			TimeMeasurementStop();
		}
//#endif AP7_PeFootRailDimensionUsed
#pragma endregion


		////////////////////////
		// 6.1 PE foot Insertion depth (above the power bar)
#pragma region
#ifdef AP7_PeFootRailDimensionUsed
		InspectionNo = 24;
		RejectStatisticGroupNo = 6;
		RejectStatisticEntryNo = 1;
		MeasurementChartNo = 20;
		if (!RejectPole() && InspectionActive(InspectionNo)&& ArticleParameter(3) != AP3_no_PE_Foot)
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("6.1 PE foot Insertion depth started", CameraNo);
			// roi defined?
			RoiName = "PeFootPresence";
			poSMs->getRoiAndParams(oResHousing, RoiName, Roi);
			if (Roi.IsEmpty())
			{
				snprintf(str_de, sizeof(str_de), "ROI '%s' nicht definiert!", RoiName);		//only ROI name
				snprintf(str_en, sizeof(str_en), "ROI '%s' isn't defined!", RoiName);		//only ROI name
				SelectLanguage(RejectMessage, str_de, str_en);
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, Roi);
			}
			else
			{
				InitSMResult(&oResPEfoot); // since ShapeModelLib V2.0.21					    
				MinValue = 50;				
				Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);
				// set ROI
				X1 = (Col1 + Col2) / 2;
				Y1 = (Row1 + Row2) / 2;
				Height = Abs(Row2 - Row1);
				Width = Abs(Col2 - Col1);

				
				ShapeModelActive = SM_PEfoot;
				int numMatches = poSMs->find(ShapeModelActive,	// name of ShapeModel
					X1, Y1,						// search position X/Y, default=0/0 for using ShapeModel settings
					Width, Height,				// search position width/helight, default=0/0 for using ShapeModel settings
					ShapeModelActivate,			// activate ShapeModel (0 for using ShapeModel settings)
					true);						// true:	ShapeModel should be found (MinValue used)
												// false:	ShapeModel should not be found (MaxValue used)
				// get ShapeModel results
				oResPEfoot = poSMs->res(ShapeModelActive,		// name of ShapeModel
					MeasurementChartNo,			// measurement chart number for score, default=-1
					MinValue, //MaxValue,		// minimum / maximum score
					Paintings,					// activate paintings, default=1
					LineWidth,					// width paintings, default=1
					0);

				// ShapeModel was found
				if (!poSMs->Err(ShapeModelActive))
				{
					peFootAngle = abs(oResPEfoot.angle_deg);
					double maxAngle = 1.2;
					if (peFootAngle > maxAngle)
					{
						snprintf(str_de, sizeof(str_de), "PE foot Error, angle = %.2f (Min %.2f)", peFootAngle, maxAngle);
						snprintf(str_en, sizeof(str_en), "PE foot Error, angle = %.2f (Max %.2f)", peFootAngle, maxAngle);
						SelectLanguage(RejectMessage, str_de, str_en);
						CHalBase::PaintRegion(Roi, WindowWithPaintings, 255, 0, 0, LineWidth);
						// set reject
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPeRailDimension, Roi);
					}

				}
				else // ShapeModel was not found
				{
					snprintf(str_de, sizeof(str_de), "PE foot Error");
					snprintf(str_en, sizeof(str_en), "PE foot Error");
					SelectLanguage(RejectMessage, str_de, str_en);
					CHalBase::PaintRegion(Roi, WindowWithPaintings, 255, 0, 0, LineWidth);
					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPeRailDimension, Roi);

				
				}

				// paintings
				Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);

				Pos.x1 = Pos.x3 = Col1;
				Pos.x2 = Pos.x4 = Col2;
				Pos.y1 = Pos.y2 = Row1;
				Pos.y3 = Pos.y4 = Row2;

				snprintf(roiName, sizeof(roiName), "%s", article.RoiPeRailFoot(1).c_str());
				bool expected = ArticleParameter(3) != AP3_no_PE_Foot;

				if ((MeasurementPoint[CameraNo][1].Result_Point_X > 0) ||
					(MeasurementPoint[CameraNo][2].Result_Point_X > 0))
				{
					double AnglePEFoot = Winkel_berechnen(MeasurementPoint[CameraNo][1].Result_Point_X, MeasurementPoint[CameraNo][1].Result_Point_Y,
						MeasurementPoint[CameraNo][2].Result_Point_X, MeasurementPoint[CameraNo][2].Result_Point_Y);

					double XMiddlePEFoot = Mittelwert(MeasurementPoint[CameraNo][1].Result_Point_X, MeasurementPoint[CameraNo][2].Result_Point_X);
					double YMiddlePEFoot = Mittelwert(MeasurementPoint[CameraNo][1].Result_Point_Y, MeasurementPoint[CameraNo][2].Result_Point_Y);

					double XPosPEFork = 0.00;
					double YPosPEFork = 0.00;

					double ProjectionAngle = AnglePEFoot - 85.869522985347530;
					ProjectionPoint_Calculate(XMiddlePEFoot, YMiddlePEFoot,		// Input Origin point [pixel]
						ProjectionAngle,								// Rotation angle [°], 0° means no rotation, psotiv angle counter clockwise
						MmToPixelDouble(4.25), MmToPixelDouble(-15.00/*15.73*/),	// [Pixel], positive values to the right / bottom
						&XPosPEFork, &YPosPEFork);						// Return values



					// Enter end point of projection line for display
					PointNo = 18;
					Set_Measurement_Point(CameraNo, PointNo, XMiddlePEFoot, YMiddlePEFoot, "MiddlePERail", 20, -1);
					//PointNo = 19;
					//Set_Measurement_Point(CameraNo, PointNo, XCorrectionPEFoot, YCorrectionPEFoot, "CorrectionPERail", 14, -1);
					PointNo = 20;
					Set_Measurement_Point(CameraNo, PointNo, XPosPEFork, YPosPEFork, "Pos.PEFork", 0, -1);

					// Calculate position filter / measurement region
					Rechteck_Eckpunkte_berechnen_Pix(CameraNo,			// Kamera Nummer
						XPosPEFork, YPosPEFork,	// Eckpunkt Rechteck [Pixel]
						0,						// Auswahl Eckpunkt Rechteck
												// 0 = Mitte
												// 1 = links oben
												// 2 = rechts oben
												// 3 = links unten
												// 4 = rechts unten
						MmToPixel(4.50),		// Breite Rechteck [Pixel]
						MmToPixel(2.25),		// Länge Rechteck [Pixel]
						AngleDeg);				// Winkel in Grad (0 = Senkrecht)

					Col1 = Pos.x1;
					Col2 = Pos.x2;
					Row1 = Pos.y1;
					Row2 = Pos.y3;
				}


				// apply special color to gray filter to remove yellow / green housing pixels
				Filter_HSL_stretching(1, CameraNo,	// Number of filter (only for display), Camera number
					Col1 - 20, Row1 - 20,			// Top left corner (Measurement area)
					Col2 + 20, Row2 + 20,			// Bottom right (Measurement area)
					0,								// Min Hue angle [0-360] (lowest border for HUE-stretching)
					360,							// Max Hue angle [0-360] (highest border for HUE-stretching)
					12,		//12						// Min Saturation [0-100] (lowest border for SAT-stretching)
					40,		//40						// Max Saturation [0-100] (highest border for SAT-stretching)
					25,		//25						// Min Brightness [0-100] (lowest border for LUM-stretching)
					90,		//90					// Max Brightness [0-100] (highest border for LUM-stretching)
					1,								// SAT Result invert. (0 = High Sat -> white Pixels, 1 = Low Sat -> white Pixels)
					1,								// 0=Add HSL-Results, 1=Multiply HSL-Results
					WindowCamera,					// Number of source image
					WindowAdditional1,				// Number of target image
					WindowAdditional1,				// Number of display window for paintings (normally the same as target image)
					1);								// Paintings

				//checkViaDeviation(oResHousing, roiName, PlcRejectPeFoot, 10, 40, 20, 70, &expected);
				// ToDo

				double r, c, angle, len1, len2;
				r = Roi.SmallestRectangle2(&c, &angle, &len1, &len2);
				HTuple rows, cols;
				VerticesRectangle2(r, c, angle, len1, len2, rows, cols);
				///*
				//				Feldfarbe(1,					// Nummer des Feldes (Anzeige)
				//					CameraNo,					// Nummer der Kamera
				//					"PE Fläche",				// Bezeichnung des Feldes
				//					Col1, Row1,				// Linke obere Ecke des Feldes
				//					Col2, Row2,				// Rechte untere Ecke des Feldes
				//					WindowAdditional1, WindowWithPaintings,			// Messbild / Anzeigebild
				//					0, 255,	// Min/Max-Wert für rote Pixel
				//					0, 255,	// Min/Max-Wert für grüne Pixel
				//					0, 255,	// Min/Max-Wert für blaue Pixel
				//					15, 255,	// Min/Max-Wert für Hell-Pixel
				//					2,					// 0 => Mittelwerte der Farben kontr.
				//												 // 1 => Pixel außerhalb der Tolerenz zählen
				//												 // 2 => Pixel innerhalb der Tolerenz zählen
				//												 // 3 => Mittelwerte der Farben innerhalb der Grenzen kontr.
				//					1);				// Felde einzeichnen (bei 1)
				//*/
				// Pixel der PE Gabeln gezielt suchen & zählen
				Feldfarbe_HSL2(1,				// Nummer des Feldes (Anzeige)
					CameraNo,					// Nummer der Kamera
					"Area PE forks",			// Bezeichnung des Feldes
					Pos.x1, Pos.y1,				// Linke obere Ecke des Feldes
					Pos.x2, Pos.y2,				// Rechte obere Ecke des Feldes
					Pos.x3, Pos.y3,				// Linke untere Ecke des Feldes
					Pos.x4, Pos.y4,				// Rechte untere Ecke des Feldes
					WindowAdditional1,			// Nummer des Messbildes
					WindowWithPaintings,		// Nummer des Anzeigebildes
					0, 360,						// Min/Max-Wert für Hue (Farbe 0-360)
					0, 100,						// Min/Max-Wert für Saturation (0-100)
					15, 100,						// Min/Max-Wert für Intensität (0-100)
					2,							// 0 => Mittelwerte der Werte ermitteln
												// 1 => Pixel außerhalb der Tolerenz zählen
												// 2 => Pixel innerhalb der Tolerenz zählen
												// 3 => Mittelwerte der Werte ermitteln (mit Abfrage Min/MaxWerte)
					1,							// passende Pixel im Feld markieren
					200, 255, 0,				// Farbe für Pixelmarkierung
					1,							// Pixel glätten (0 - 8)
					0,							// Pixel außerhalb der Toleranz einzeichnen (nur Modus 2)
					1);							// Feld einzeichnen (ab 1)
				Value = PixelToMm2(FeldfarbwertHSL.ZaehlerInnerhalb);
				// 2024-11-13
				MinValue = 2.12;//1.7 2.29
				MaxValue = 3.50;//3.5

				// check result
				if (Value < MinValue)
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Fläche PE Gabeln = %.2f (Min %.2f)", Value, MinValue);
					snprintf(str_en, sizeof(str_en), "Area PE forks = %.2f (Min %.2f)", Value, MinValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPeRailDimension, Pos.x1, Pos.y1, Pos.x4, Pos.y4);
				}

				// check result
				if (Value > MaxValue)
				{
					// set reject message
					snprintf(str_de, sizeof(str_de), "Fläche PE Gabeln = %.2f (Max %.2f)", Value, MaxValue);
					snprintf(str_en, sizeof(str_en), "Area PE forks = %.2f (Max %.2f)", Value, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);

					// set reject
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectPeRailDimension, Pos.x1, Pos.y1, Pos.x4, Pos.y4);
				}

				// set data chart
				MeasurementChartSet(MeasurementChartNo, Value, MinValue, MaxValue, MinValue - 1.50, MaxValue + 1.50);
			}
			Logger::getInstance().LogInfo("6.1 PE foot Insertion depth stop", CameraNo);
			TimeMeasurementStop();
		}
#endif AP7_PeFootRailDimensionUsed
#pragma endregion

		//////////////////////
	//6.8 PE foot check
	//No PeFoot check 
#pragma region STDC_000_TemplateStandardInspection
		InspectionNo = 33;
		RejectStatisticGroupNo = 6;
		RejectStatisticEntryNo = 8;
		MeasurementChartNo = 10;
		Paintings = PaintingsEditable2();
		if (!RejectPole() && InspectionActive(InspectionNo) && ArticleParameter(3) == AP3_no_PE_Foot)
		{
			TimeMeasurementStart();
			Logger::getInstance().LogInfo("6.8 no pe foot check started", CameraNo);
			HTuple roiIndex, infoParams;
			const char* peFootName;
			articleDef.FindRegionNum(roiIndex, oResHousing, "NoPEFoot");
			for (int i = 0; i < roiIndex.Num(); i++)
			{
				peFootName = articleDef.GetRoiByFuzzyRegion(roiIndex, oResHousing, i, Roi, poSMs, infoParams);
				//DumpTuple(infoParams);
				MaxValue = stod(infoParams[0].S());
				// check roi
				if (Roi.IsEmpty())
				{
					snprintf(str_de, sizeof(str_de), "ROI '%s' nicht definiert!", RoiName);
					snprintf(str_en, sizeof(str_en), "ROI '%s' isn't defined!", RoiName);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectMalfunction, Roi);
				}
				else
				{
					HImage hiPImgPeFoot = poHImg->getImg()->ReduceDomain(Roi).Rgb1ToGray();
					if (Paintings)
					{
						CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 0, 0, LineWidth * 2); 	// shadow
						CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, LineWidth); 	// automatically red painted in case of reject
					}

					double deviation = 0.0;
					double intensity = 0.0;
					intensity = Roi.Intensity(hiPImgPeFoot, &deviation);
					Value = intensity;	// result

					int textAlignmentX = eAlignCenter;
					int textAlignmentY = eAlignBottom;
					Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);
					Y1 = Row1 - MmToPixel(0.50);
					snprintf(str_de, sizeof(str_de), "PEfoot = %.2f (Max %.2f)", Value, MaxValue);
					SelectLanguage(string1, str_de, str_en);
					CHalBase::PaintText(string1, Col1 / 2 + Col2 / 2, Y1, WindowWithPaintings,0, 255, 255, FontSize+10, HW_FONT_ARIAL, false, false, 0, textAlignmentX, textAlignmentY, false);


					// check result
					if (Value > MaxValue)
					{
						// set reject message
						snprintf(str_de, sizeof(str_de), "PE foot Error, Streuung = %.2f (Min %.2f)", Value, MaxValue);
						snprintf(str_en, sizeof(str_en), "PE foot Error, intensity = %.2f (Max %.2f)", Value, MaxValue);
						SelectLanguage(RejectMessage, str_de, str_en);
						CHalBase::PaintRegion(Roi, WindowWithPaintings, 255, 0, 0, LineWidth);
						// set reject
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, PlcRejectArticleChangeFailed, Roi);
					}



				}

#pragma endregion outside STDC_000_TemplateStandardInspection
			}//for
			Logger::getInstance().LogInfo("6.8 no pe foot check stop, CameraNo");
			TimeMeasurementStop();
		}
#pragma endregion

		// Display results of all measurement points
		Zeit_Start();
		bool ColorValuesOK = TRUE;
		Display_Measurement_Points(CameraNo, WindowWithPaintings, PaintingsEditable4(), (int)(2.00 * FontSize), 255, 0, 255, ColorValuesOK);
		Zeit_Ende(CameraNo, "Display Meas.Points");
	

		//---------------------------  End of pole check  ----------------------------------

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


bool checkViaDeviation(HImage HImg,								// HImage auf dem gearbeitet wird
	char smName[50],											// Name of the model (xxx)
	SMLibResult smResult,										// Result of the shape model search (for position, angle, ...)
	char roiName[50],											// Name of region to check deviation ("MoldCheckx")
	const BYTE plcResult,										// in case of error -> number for return to PLC
	double paramMinDev_MinValue, double paramMinDev_MaxValue,	// MIN Deviation: Erlaubter Bereich (Min, Max)
	double paramMaxDev_MinValue, double paramMaxDev_MaxValue,	// MAX Deviation: Erlaubter Bereich (Min, Max) für MAX Deviation
	bool* expected,												// set to 'nullptr'
	int ModelIndex,												// Model Index für das Ziel Modell (bei max 8: 0 - 7)
	int LocalMeasurementChartNo,								// WerBereich Nummer für das Ergebnis der Streuung (only MC 1-6 reserved)
	int LocicalUnderfilledNumber)								// Loigsche Nummer der Unterspritzung an diesem Artikel (fortlaufend von 1 - 6 max.)
{
	HTuple params;

	double localMin = paramMinDev_MinValue;
	if (localMin == 0)
	{
		localMin = Hidden;
	}

	int InfoParamNo = 1;
	int CheckMethod = 0;		// Check by deviation

	// get roi
	poSMs->getRoiAndParams(smResult, roiName, Roi);
	if (Roi.IsEmpty())
	{
		snprintf(str_de, sizeof(str_de), "ROI '%s' nicht definiert!", roiName);
		snprintf(str_en, sizeof(str_en), "ROI '%s' not defined!", roiName);
		SelectLanguage(RejectMessage, str_de, str_en);
		RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
	}
	else
	{
		// get roi info parameters
		//HTuple roiInfoParams = poSMs->getRoiNameAndInfo(article.ShapeArticleName(), 0, roiName, 0);
		HTuple roiInfoParams = poSMs->getRoiNameAndInfo(smName, ModelIndex, roiName, 0);
		DumpTuple(roiInfoParams);
		if (expected != nullptr)
		{
			// use "expected"
			if (HTuple(roiInfoParams[InfoParamNo]).Strlen() < 11)
			{
				snprintf(str_de, sizeof(str_de), "ROI '%s' erfordert 'Info'-Parameter: 'un-/expected;[min];[max]'!", roiName);
				snprintf(str_en, sizeof(str_en), "ROI '%s' requires 'Info' parameters: 'un-/expected;[min];[max]'!", roiName);
				SelectLanguage(RejectMessage, str_de, str_en);
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
			}
			else
			{
				params = HTuple(roiInfoParams[InfoParamNo]).Split(";");
				if ((params.Num() != 4) &&	// Deviation: 'Dev;un-/expected;[min];[max]'
					(params.Num() != 6))	// Color 'Color;un-/expected;Range Hue[%%];Range Sat[%%];Range Lum[%%];DefectSize[mm])'
				{
					snprintf(str_de, sizeof(str_de), "ROI '%s' erfordert 4/6 'Info'-Parameter: 'un-/expected;[min];[max]'!", roiName);
					snprintf(str_en, sizeof(str_en), "ROI '%s' requires 4/6 'Info' parameters: 'un-/expected;[min];[max]'!", roiName);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
				}
				if (params.Num() == 4)
				{
					if (strstr(roiInfoParams[InfoParamNo], "Dev") == 0)
					{
						snprintf(str_de, sizeof(str_de), "ROI '%s' erfordert 4 'Info'-Parameter: 'Dev;un-/expected;[min];[max]'!", roiName);
						snprintf(str_en, sizeof(str_en), "ROI '%s' requires 3 'Info' parameters: 'Dev;un-/expected;[min];[max]'!", roiName);
						SelectLanguage(RejectMessage, str_de, str_en);
						RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
					}
				}
			}
		}
		else
		{
			// ignore "expected"
			if (HTuple(roiInfoParams[InfoParamNo]).Strlen() < 3)
			{
				snprintf(str_de, sizeof(str_de), "ROI '%s' erfordert 'Info'-Parameter: '[min];[max]'!", roiName);
				snprintf(str_en, sizeof(str_en), "ROI '%s' requires 'Info' parameters: '[min];[max]'!", roiName);
				SelectLanguage(RejectMessage, str_de, str_en);
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
			}
			else
			{
				params = HTuple(roiInfoParams[InfoParamNo]).Split(";");
				if ((params.Num() != 3) &&	// Deviation: 'Dev;un-/expected;[min];[max]'
					(params.Num() != 5))	// Color 'Color;un-/expected;Range Hue[%%];Range Sat[%%];Range Lum[%%];DefectSize[mm])'
				{
					snprintf(str_de, sizeof(str_de), "ROI '%s' erfordert 3 oder 5 'Info'-Parameter: 'Dev;[min];[max] / Color;[H%%];[S%%];[L%%];[ErrorSize MM]'!", roiName);
					snprintf(str_en, sizeof(str_en), "ROI '%s' requires 3 or 5 'Info' parameters: 'Dev;[min];[max] / Color;[H%%];[S%%];[L%%];[ErrorSize MM]'!", roiName);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
				}
				else
				{
					if (strstr(roiInfoParams[InfoParamNo], "Dev") > 0)
					{
						if (params.Num() != 3)
						{
							snprintf(str_de, sizeof(str_de), "ROI '%s' erfordert 3 'Info'-Parameter: 'Dev;[min];[max]'!", roiName);
							snprintf(str_en, sizeof(str_en), "ROI '%s' requires 3 'Info' parameters: 'Dev;[min];[max]'!", roiName);
							SelectLanguage(RejectMessage, str_de, str_en);
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
						}
					}
					if (strstr(roiInfoParams[InfoParamNo], "Color") > 0)
					{
						if (params.Num() != 5)
						{
							snprintf(str_de, sizeof(str_de), "ROI '%s' erfordert 5 'Info'-Parameter: 'Color;[H%%];[S%%];[L%%];[ErrorSize MM]'!", roiName);
							snprintf(str_en, sizeof(str_en), "ROI '%s' requires 5 'Info' parameters: 'Color;[H%%];[S%%];[L%%];[ErrorSize MM]'!", roiName);
							SelectLanguage(RejectMessage, str_de, str_en);
							RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
						}
					}
				}
			}
		}
	}

//	if (!RejectPole())
	{
		// check info parameter values
		int ParamNo = 0;
		int FirstParamNo = 0;
		for (int i = 0; /*!RejectPole() &&*/ i < params.Num(); i++)
		{
			// un-/expected
			if (expected != nullptr && i == 0)
			{
				string value = params[i].S();
				std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });	// too lower case
				bool expectedPredfined = *expected;
				if (value == "expected")
				{
					FirstParamNo++;
					*expected = true;
				}
				else if (value == "unexpected")
				{
					ParamNo++;
					*expected = false;
				}
				else
				{
					snprintf(str_de, sizeof(str_de), "ROI '%s': 'expected' oder 'unexpected' als erster Info-Parameter erwartet!", roiName);
					snprintf(str_en, sizeof(str_en), "ROI '%s': 'expected' or 'unexpected' required as first info parameter!", roiName);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
					continue;
				}
				// predefined expectations
				if (expectedPredfined != *expected)
				{
					snprintf(str_de, sizeof(str_de), "'%s' laut Artikelparameter %s, laut Info-Parameter %s", roiName, expectedPredfined ? "erwartet" : "nicht erwartet", *expected ? "erwartet" : "nicht erwartet");
					snprintf(str_en, sizeof(str_en), "'%s' according to article parameters %s, according to info parameter %s", roiName, expectedPredfined ? "expected" : "unexpected", *expected ? "expected" : "unexpected");
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
					continue;
				}
				FirstParamNo++;
			}

			// Methode abfragen
			if (ParamNo == FirstParamNo)
			{
				string value = params[i].S();
				if (value == "Color")
				{
					CheckMethod = 1;
				}
			}

			// Min deviation
			if (ParamNo == FirstParamNo + 1)
			{
				float value = stof(params[i].S());
				MinValue = paramMinDev_MinValue;
				MaxValue = paramMinDev_MaxValue;
				if (value < MinValue || value > MaxValue)
				{
					snprintf(str_de, sizeof(str_de), "ROI '%s' Info-Parameter 'min' Wert: %.0f (Min %.0f, Max %.0f)!", roiName, value, MinValue, MaxValue);
					snprintf(str_en, sizeof(str_en), "ROI '%s' info parameter 'min' value: %.0f (Min %.0f, Max %.0f)!", roiName, value, MinValue, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
					continue;
				}
				Value = value;	// buffer
			}

			// Max deviation
			if (ParamNo == FirstParamNo + 2)
			{
				float value = stof(params[i].S());
				MinValue = paramMaxDev_MinValue;
				MaxValue = paramMaxDev_MaxValue;
				if (value < MinValue || value > MaxValue)
				{
					snprintf(str_de, sizeof(str_de), "ROI '%s' Info-Parameter 'max' Wert: %.0f (Min %.0f, Max %.0f)!", roiName, value, MinValue, MaxValue);
					snprintf(str_en, sizeof(str_en), "ROI '%s' info parameter 'max' value: %.0f (Min %.0f, Max %.0f)!", roiName, value, MinValue, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
					continue;
				}
				MinValue = Value;
				MaxValue = value;
				if (MinValue > MaxValue)
				{
					snprintf(str_de, sizeof(str_de), "ROI '%s' Info-Parameter 'min'(%.0f) < 'max' (%.0f) erwartet!", roiName, MinValue, MaxValue);
					snprintf(str_en, sizeof(str_en), "ROI '%s' info parameter 'min'(%.0f) < 'max' (%.0f) expected!", roiName, MinValue, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
					continue;
				}
			}

			ParamNo++;
		}
	}

	// check deviation
	if (1)//(!RejectPole())
	{

		Roi.Intensity(HImg.ReduceDomain(Roi).Rgb1ToGray(), &Value);

		if (expected != nullptr)
		{
			// expected
			if (*expected)
			{
				if (Value < MinValue || Value > MaxValue)
				{
					snprintf(str_de, sizeof(str_de), "'%s' nicht gefunden: %.2f (Min %.0f, Max %.0f)", roiName, Value, MinValue, MaxValue);
					snprintf(str_en, sizeof(str_en), "Expected '%s' missing: %.2f (Min %.0f, Max %.0f)", roiName, Value, MinValue, MaxValue);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
				}

				// set data chart
				MeasurementChartSet(LocalMeasurementChartNo, Value, localMin, MaxValue, 0, MaxValue + 5.0);
			}
			// unexpected
			else
			{
				if (Value > MinValue)
				{
					snprintf(str_de, sizeof(str_de), "'%s' gefunden: %.2f (Max %.0f)", roiName, Value, MinValue);
					snprintf(str_en, sizeof(str_en), "Unexpected '%s' found: %.2f (Max %.0f)", roiName, Value, MinValue);
					SelectLanguage(RejectMessage, str_de, str_en);
					RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
				}

				// set data chart
				MeasurementChartSet(LocalMeasurementChartNo, Value, localMin, MinValue, 0, MinValue + 5.0);
			}
		}
		else
		{
			if (Value < MinValue || Value > MaxValue)
			{
				snprintf(str_de, sizeof(str_de), "%s: %.2f (Min %.0f, Max %.0f)", roiName, Value, MinValue, MaxValue);
				snprintf(str_en, sizeof(str_en), "%s: %.2f (Min %.0f, Max %.0f)", roiName, Value, MinValue, MaxValue);
				SelectLanguage(RejectMessage, str_de, str_en);
				RejectSet(RejectStatisticGroupNo, RejectStatisticEntryNo, RejectMessage, plcResult, Roi);
			}

			// set data chart
			MeasurementChartSet(LocalMeasurementChartNo, Value, localMin, MaxValue, 0, MaxValue + 5.0);
		}

		// paintings
		if (!Reject())  CHalBase::PaintRegion(Roi, WindowWithPaintings, 0, 250, 250, 1);
		else			CHalBase::PaintRegion(Roi, WindowWithPaintings, R, G, B, 1);
		snprintf(string1, sizeof(string1), "%i:%s", LocicalUnderfilledNumber, roiName);
		if (expected != nullptr)
		{
			snprintf(str_de, sizeof(str_de), "%s %s", *expected ? "Erwartet" : "Unerwartet", roiName);
			snprintf(str_en, sizeof(str_en), "%s %s", *expected ? "Expected" : "Unexpected", roiName);
			SelectLanguage(string1, str_de, str_en);
		}
	}
	else
		snprintf(string1, sizeof(string1), "%i:%s", LocicalUnderfilledNumber, roiName);

	// paintings
	Row1 = Roi.SmallestRectangle1(&Col1, &Row2, &Col2);
	//Width = Col2 - Col1;
	Height = Row2 - Row1;
	CenterRow = Row1 + Height * 0.5;
	//CenterCol = Col1 + Width * 0.5;

//	if (Paintings > 0)
		CHalBase::PaintText(string1, Col2, Row1 - FontSize, WindowWithPaintings, R, G, B, FontSize, HW_FONT_ARIAL, true, false, 0, eAlignRight, eAlignCenter, true);

	return !RejectPole();
}
