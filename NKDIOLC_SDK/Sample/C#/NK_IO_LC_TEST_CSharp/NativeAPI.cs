using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace NK_IO_LC_TEST_CSharp
{
    public struct OPENCOM_CALLBACK_ARG_T
    {
        public UInt16 portNum;
        public Byte devId;
        public Byte hardwareMajorVer;
        public Byte hardwareMinorVer;
        public Byte hardwareRevVer;
        public Byte firmwareMajorVer;
        public Byte firmwareMinorVer;
        public Byte firmwareRevVer;
        public Byte fillup_1;
        public Byte fillup_2;
        public Byte error;
        public UInt32 errorId;
    }

	public struct CLOSECOM_CALLBACK_ARG_T
    {
        public UInt32 fill_up;
        public Byte fill_up_1;
        public Byte fill_up_2;
        public Byte fill_up_3;
        public Byte error;
        public UInt32 errorId;
        public UInt32 fill_up_4;
    }

	public struct GET_DEVICE_VER_CALLBACK_ARG_T
    {
        public Byte hardwareMajorVer;
        public Byte hardwareMinorVer;
        public Byte hardwareRevVer;
        public Byte firmwareMajorVer;
        public Byte firmwareMinorVer;
        public Byte firmwareRevVer;
        public Byte fill_up;
        public Byte error;
        public UInt32 errorId;
        public UInt32 fill_up_2;
    }

	public struct SET_PWM_PARAMS_CALLBACK_ARG_T
    {
        public UInt32 fill_up;
        public Byte chIdx;
        public Byte fill_up_1;
        public Byte fill_up_2;
        public Byte error;
        public UInt32 errorId;
        public UInt32 fill_up_3;
    }

	public struct GET_PWM_PARAMS_CALLBACK_ARG_T
    {
        public Byte chIdx;
        public Byte pwmMode;
        public Byte pwmValue;
        public Byte pwmHoldingTime;
        public Byte pwmOnOff;
        public Byte fill_up_2;
        public Byte fill_up_3;
        public Byte error;
        public UInt32 errorId;
        public UInt32 fill_up_4;
    }

    public struct SET_GENERAL_PARAM_CALLBACK_ARG_T
    {
        public Byte devId;
        public Byte cmdId;
        public Byte paramId;
        public Byte paramLen;
        public UInt32 paramValue;
        public Byte fill_up_1;
        public Byte fill_up_2;
        public Byte fill_up_3;
        public Byte error;
        public UInt32 errorId;
    }
    

	public struct GET_GENERAL_PARAM_CALLBACK_ARG_T
    {
        public Byte devId;
        public Byte cmdId;
        public Byte paramId;
        public Byte paramLen;
        public UInt32 paramValue;
        public Byte fill_up_1;
        public Byte fill_up_2;
        public Byte fill_up_3;
        public Byte error;
        public UInt32 errorId;
    }
    


    [StructLayout(LayoutKind.Explicit)]
    public class LC_CALLBACK_ARG_T
    {
        [FieldOffset(0)]
        public OPENCOM_CALLBACK_ARG_T openComCallbackArg;
        [FieldOffset(0)]
        public CLOSECOM_CALLBACK_ARG_T closeComCallbackArg;
        [FieldOffset(0)]
        public GET_DEVICE_VER_CALLBACK_ARG_T getDeviceVerCallbackArg;
        [FieldOffset(0)]
        public SET_PWM_PARAMS_CALLBACK_ARG_T setPwmParamsCallbackArg;
        [FieldOffset(0)]
        public GET_PWM_PARAMS_CALLBACK_ARG_T getPwmParamsCallbackArg;
        [FieldOffset(0)]
        public SET_GENERAL_PARAM_CALLBACK_ARG_T setGeneralParamCallbackArg;
        [FieldOffset(0)]
        public GET_GENERAL_PARAM_CALLBACK_ARG_T getGeneralParamCallbackArg;

    };

    // Light Control
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void LCCallbackMethod(LC_CALLBACK_ARG_T arg);//委托类型，主要用于回调。

    public static class NativeAPI
    {

       

        // Library
        [DllImport("NKIOLIBx64.dll", EntryPoint = "NKDIO_LibraryInit", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKDIO_LibraryInit(string configIniFile);
        [DllImport("NKIOLIBx64.dll", EntryPoint = "NKDIO_LibraryDeinit", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NKDIO_LibraryDeinit();

        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_LibraryInit", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_LibraryInit(string configIniFile);
        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_LibraryDeinit", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_LibraryDeinit();

        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_OpenDevice_Async", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_OpenDevice_Async(UInt16 port, LCCallbackMethod pCallBackFun);
        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_CloseDevice_Async", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_CloseDevice_Async(UInt16 port, LCCallbackMethod pCallBackFun);
        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_IsDeviceOpened_Async", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_IsDeviceOpened_Async(UInt16 port, UInt32 devId);
        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_Process_Async", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_Process_Async();



        // Read and Write DIO in polling mode
        /*-Polling Mode API------------------------------------------------------------------------------*/
        [DllImport("NKIOLIBx64.dll", EntryPoint = "NKDIO_PollingReadDiByte", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKDIO_PollingReadDiByte(Byte diByteIndex, Byte[] pByteValue);
        [DllImport("NKIOLIBx64.dll", EntryPoint = "NKDIO_PollingWriteDoByte", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKDIO_PollingWriteDoByte(Byte doByteIndex, Byte doByteValue);



        /*-Light control API------------------------------------------------------------------------------*/
        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_GetVerInfo_Async", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_GetVerInfo_Async(UInt32 devId, LCCallbackMethod pCallBackFun);

        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_SetPwmParams_Async", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_SetPwmParams_Async(UInt32 devId,
            Byte ucChIdx,
            Byte ucPwmMode,
            Byte ucPwmValue,
            Byte ucPwmHoldingTime,
            Byte ucPwmOnOff,
            LCCallbackMethod pCallBackFun);

        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_GetPwmParams_Async", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_GetPwmParams_Async(UInt32 devId,
            Byte ucChIdx,
            LCCallbackMethod pCallBackFun);

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Add for the parameters settings
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// For the general parameter settings
        /// </summary>
        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_SetGeneralParam_Async", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_SetGeneralParam_Async(UInt32 devId,
            Byte ucParamId,
            Byte ucParamLen,
            UInt32 uiParamValue,
            LCCallbackMethod pCallBackFun);

        [DllImport("NKLCLIBx64.dll", EntryPoint = "NKLC_GetGeneralParam_Async", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern Int32 NKLC_GetGeneralParam_Async(UInt32 devId,
            Byte ucParamId,
            Byte ucParamLen,
            LCCallbackMethod pCallBackFun);

    }
}
