using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Text.RegularExpressions;



namespace NK_IO_LC_TEST_CSharp
{


    public partial class Form1 : Form
    {
        public static Form1 pThis;

        private int m_dioTestStartFlag = 0;

        //private UInt16 m_doValue = 0;
        private Byte m_doValueL = 0;
        private Byte m_doValueH = 0;
        private Byte[] m_diValueH = new byte[1];
        private Byte[] m_diValueL = new byte[1];

        private UInt16 m_comPort = 3;
        private Int32 m_comPortOpenFlag = 0;
        private Byte m_DevId = 0x01;

        private Byte m_pwmValueCh0 = 0;
        private Byte m_pwmValueCh1 = 0;
        private Byte m_pwmValueCh2 = 0;
        private Byte m_pwmValueCh3 = 0;

        string configPath = "";

        private LCCallbackMethod pvOpenComCallback = openComCallBcak;
        private LCCallbackMethod pvCloseComCallback = closeComCallBack;
        private LCCallbackMethod pvGetDeviceVerCallback = getDeviceVerCallBack;

        private LCCallbackMethod pvGetCh0PwmParamsCallback = getPwmParamsCallBackCh0;
        private LCCallbackMethod pvSetCh0PwmParamsCallback = setPwmParamsCallBackCh0;

        private LCCallbackMethod pvGetCh1PwmParamsCallback = getPwmParamsCallBackCh1;
        private LCCallbackMethod pvSetCh1PwmParamsCallback = setPwmParamsCallBackCh1;

        private LCCallbackMethod pvGetCh2PwmParamsCallback = getPwmParamsCallBackCh2;
        private LCCallbackMethod pvSetCh2PwmParamsCallback = setPwmParamsCallBackCh2;

        private LCCallbackMethod pvGetCh3PwmParamsCallback = getPwmParamsCallBackCh3;
        private LCCallbackMethod pvSetCh3PwmParamsCallback = setPwmParamsCallBackCh3;

        public Form1()
        {
            InitializeComponent();
            pThis = this;
            this.m_comboBoxModeCh0.SelectedIndex = 0;
            this.m_comboBoxModeCh1.SelectedIndex = 0;
            this.m_comboBoxModeCh2.SelectedIndex = 0;
            this.m_comboBoxModeCh3.SelectedIndex = 0;

            this.m_comboBoxPort.Items.Add("COM3");
            this.m_comboBoxPort.SelectedItem = "COM3";

            string selectFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "select.ini");
            string selectItem = INIHelper.Read("SELECTED", "Name", " ", selectFilePath) ;
            string selectItemPath = INIHelper.Read("SELECTED", "ConfigPath", " ", selectFilePath);
            configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, selectItem);
            configPath = configPath + "\\nkio_config.ini";

            if (NativeAPI.NKDIO_LibraryInit(configPath) != 0)
            {
                MessageBox.Show("DIOLC_LibraryBaseInit error");
                return;
            }
            if (NativeAPI.NKLC_LibraryInit(configPath) != 0)
            {
                MessageBox.Show("DIOLC_LibraryBaseInit error");
                return;
            }

            //this.labelTest.Text = configPath;
            this.timer1.Enabled = true;

        }

        private void m_checkBoxDO_CheckedChanged(object sender, EventArgs e)
        {
            if (m_dioTestStartFlag == 1)
            {
                Byte flag = 0x01;
                if (m_checkBoxDO0.Checked)
                {
                    flag = 0x01;
                    flag = (Byte)(~flag);
                    m_doValueL &= flag;
                }
                else
                {
                    flag = 0x01;
                    m_doValueL |= flag;
                }

                if (m_checkBoxDO1.Checked)
                {
                    flag = 0x02;
                    flag = (Byte)(~flag);
                    m_doValueL &= flag;

                }
                else
                {
                    flag = 0x02;
                    m_doValueL |= flag;
                }

                if (m_checkBoxDO2.Checked)
                {
                    flag = 0x04;
                    flag = (Byte)(~flag);
                    m_doValueL &= flag;
                }
                else
                {
                    flag = 0x04;
                    m_doValueL |= flag;

                }

                if (m_checkBoxDO3.Checked)
                {
                    flag = 0x08;
                    flag = (Byte)(~flag);
                    m_doValueL &= flag;
                }
                else
                {
                    flag = 0x08;
                    m_doValueL |= flag;

                }

                if (m_checkBoxDO4.Checked)
                {
                    flag = 0x10;
                    flag = (Byte)(~flag);
                    m_doValueL &= flag;
                }
                else
                {

                    flag = 0x10;
                    m_doValueL |= flag;
                }

                if (m_checkBoxDO5.Checked)
                {
                    flag = 0x20;
                    flag = (Byte)(~flag);
                    m_doValueL &= flag;
                }
                else
                {
                    flag = 0x20;
                    m_doValueL |= flag;

                }

                if (m_checkBoxDO6.Checked)
                {
                    flag = 0x40;
                    flag = (Byte)(~flag);
                    m_doValueL &= flag;
                }
                else
                {
                    flag = 0x40;
                    m_doValueL |= flag;

                }

                if (m_checkBoxDO7.Checked)
                {
                    flag = 0x80;
                    flag = (Byte)(~flag);
                    m_doValueL &= flag;
                }
                else
                {
                    flag = 0x80;
                    m_doValueL |= flag;

                }

                //
                if (m_checkBoxDO8.Checked)
                {
                    flag = 0x01;
                    flag = (Byte)(~flag);
                    m_doValueH &= flag;
                }
                else
                {
                    flag = 0x01;
                    m_doValueH |= flag;
                }

                if (m_checkBoxDO9.Checked)
                {
                    flag = 0x02;
                    flag = (Byte)(~flag);
                    m_doValueH &= flag;

                }
                else
                {
                    flag = 0x02;
                    m_doValueH |= flag;
                }

                if (m_checkBoxDO10.Checked)
                {
                    flag = 0x04;
                    flag = (Byte)(~flag);
                    m_doValueH &= flag;
                }
                else
                {
                    flag = 0x04;
                    m_doValueH |= flag;

                }

                if (m_checkBoxDO11.Checked)
                {
                    flag = 0x08;
                    flag = (Byte)(~flag);
                    m_doValueH &= flag;
                }
                else
                {
                    flag = 0x08;
                    m_doValueH |= flag;

                }

                if (m_checkBoxDO12.Checked)
                {
                    flag = 0x10;
                    flag = (Byte)(~flag);
                    m_doValueH &= flag;
                }
                else
                {

                    flag = 0x10;
                    m_doValueH |= flag;
                }

                if (m_checkBoxDO13.Checked)
                {
                    flag = 0x20;
                    flag = (Byte)(~flag);
                    m_doValueH &= flag;
                }
                else
                {
                    flag = 0x20;
                    m_doValueH |= flag;

                }

                if (m_checkBoxDO14.Checked)
                {
                    flag = 0x40;
                    flag = (Byte)(~flag);
                    m_doValueH &= flag;
                }
                else
                {
                    flag = 0x40;
                    m_doValueH |= flag;

                }

                if (m_checkBoxDO15.Checked)
                {
                    flag = 0x80;
                    flag = (Byte)(~flag);
                    m_doValueH &= flag;
                }
                else
                {
                    flag = 0x80;
                    m_doValueH |= flag;

                }

                NativeAPI.NKDIO_PollingWriteDoByte(0, (Byte)m_doValueL);
                NativeAPI.NKDIO_PollingWriteDoByte(1, (Byte)m_doValueH);
            }
            else
            {

            }
        }

        private void m_checkBoxStartDoTest_CheckedChanged(object sender, EventArgs e)
        {
            if (m_checkBoxStartDoTest.Checked)
            {
                m_checkBoxDO0.Checked = false;
                m_checkBoxDO1.Checked = false;
                m_checkBoxDO2.Checked = false;
                m_checkBoxDO3.Checked = false;
                m_checkBoxDO4.Checked = false;
                m_checkBoxDO5.Checked = false;
                m_checkBoxDO6.Checked = false;
                m_checkBoxDO7.Checked = false;
                m_checkBoxDO8.Checked = false;
                m_checkBoxDO9.Checked = false;
                m_checkBoxDO10.Checked = false;
                m_checkBoxDO11.Checked = false;
                m_checkBoxDO12.Checked = false;
                m_checkBoxDO13.Checked = false;
                m_checkBoxDO14.Checked = false;
                m_checkBoxDO15.Checked = false;

               
                m_dioTestStartFlag = 1;

            }
            else
            {
                m_dioTestStartFlag = 0;
            }
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
          
            // Notification Process
             NativeAPI.NKLC_Process_Async();
          
            
            if (m_dioTestStartFlag == 1)
            {
                NativeAPI.NKDIO_PollingReadDiByte(0, m_diValueL);
                NativeAPI.NKDIO_PollingReadDiByte(1, m_diValueH);

                if ((m_diValueL[0] & 0x01) == 0x01)
                {
                    m_lblDI0.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI0.BackColor = Color.Gray;
                }

                if ((m_diValueL[0] & 0x02) == 0x02)
                {
                    m_lblDI1.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI1.BackColor = Color.Gray;
                }

                if ((m_diValueL[0] & 0x04) == 0x04)
                {
                    m_lblDI2.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI2.BackColor = Color.Gray;
                }

                if ((m_diValueL[0] & 0x08) == 0x08)
                {
                    m_lblDI3.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI3.BackColor = Color.Gray;
                }

                if ((m_diValueL[0] & 0x10) == 0x10)
                {
                    m_lblDI4.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI4.BackColor = Color.Gray;
                }

                if ((m_diValueL[0] & 0x20) == 0x20)
                {
                    m_lblDI5.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI5.BackColor = Color.Gray;
                }

                if ((m_diValueL[0] & 0x40) == 0x40)
                {
                    m_lblDI6.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI6.BackColor = Color.Gray;
                }

                if ((m_diValueL[0] & 0x80) == 0x80)
                {
                    m_lblDI7.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI7.BackColor = Color.Gray;
                }

                // di8~15
                if ((m_diValueH[0] & 0x01) == 0x01)
                {
                    m_lblDI8.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI8.BackColor = Color.Gray;
                }

                if ((m_diValueH[0] & 0x02) == 0x02)
                {
                    m_lblDI9.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI9.BackColor = Color.Gray;
                }

                if ((m_diValueH[0] & 0x04) == 0x04)
                {
                    m_lblDI10.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI10.BackColor = Color.Gray;
                }

                if ((m_diValueH[0] & 0x08) == 0x08)
                {
                    m_lblDI11.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI11.BackColor = Color.Gray;
                }

                if ((m_diValueH[0] & 0x10) == 0x10)
                {
                    m_lblDI12.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI12.BackColor = Color.Gray;
                }

                if ((m_diValueH[0] & 0x20) == 0x20)
                {
                    m_lblDI13.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI13.BackColor = Color.Gray;
                }

                if ((m_diValueH[0] & 0x40) == 0x40)
                {
                    m_lblDI14.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI14.BackColor = Color.Gray;
                }

                if ((m_diValueH[0] & 0x80) == 0x80)
                {
                    m_lblDI15.BackColor = Color.LimeGreen;
                }
                else
                {
                    m_lblDI15.BackColor = Color.Gray;
                }


            }
            else
            {
                m_lblDI0.BackColor = Color.Gray;
                m_lblDI1.BackColor = Color.Gray;
                m_lblDI2.BackColor = Color.Gray;
                m_lblDI3.BackColor = Color.Gray;
                m_lblDI4.BackColor = Color.Gray;
                m_lblDI5.BackColor = Color.Gray;
                m_lblDI6.BackColor = Color.Gray;
                m_lblDI7.BackColor = Color.Gray;
                m_lblDI8.BackColor = Color.Gray;
                m_lblDI9.BackColor = Color.Gray;
                m_lblDI10.BackColor = Color.Gray;
                m_lblDI11.BackColor = Color.Gray;
                m_lblDI12.BackColor = Color.Gray;
                m_lblDI13.BackColor = Color.Gray;
                m_lblDI14.BackColor = Color.Gray;
                m_lblDI15.BackColor = Color.Gray;
            }
        }

        // Light Control Page
        private void m_btnConnect_Click(object sender, EventArgs e)
        {
            string szComSelected = this.m_comboBoxPort.Text;
            if (szComSelected != null && szComSelected != string.Empty)
            {
                // 正则表达式剔除非数字字符（不包含小数点.）
                szComSelected = Regex.Replace(szComSelected, @"[^\d.\d]", "");
                // 如果是数字，则转换为decimal类型
                if (Regex.IsMatch(szComSelected, @"^[+-]?\d*[.]?\d*$"))
                {
                    m_comPort = UInt16.Parse(szComSelected);
                }
            }

            if (m_comPortOpenFlag == 0)
            {
                NativeAPI.NKLC_OpenDevice_Async( m_comPort, this.pvOpenComCallback);
               
                m_comPortOpenFlag = 1;
                
            }
            else
            {
                NativeAPI.NKLC_CloseDevice_Async( m_comPort, this.pvCloseComCallback);
                m_comPortOpenFlag = 0;
            }
        }


        private void m_btnReadCh0_Click(object sender, EventArgs e)
        {
            NativeAPI.NKLC_GetPwmParams_Async(m_DevId, 0x01, this.pvGetCh0PwmParamsCallback);
        }

        private void m_btnWriteCh0_Click(object sender, EventArgs e)
        {
            NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                    0x01,
                    (byte)this.m_comboBoxModeCh0.SelectedIndex,
                    m_pwmValueCh0,
                    (byte)this.m_HoldingTimeCh0.Value,
                    m_checkBoxSwitchCh0.Checked ? (byte)0x01:(byte)0,
                    this.pvSetCh0PwmParamsCallback);
        }

        private void m_btnReadCh1_Click(object sender, EventArgs e)
        {
            NativeAPI.NKLC_GetPwmParams_Async(m_DevId, 0x02, this.pvGetCh1PwmParamsCallback);
        }

        private void m_btnWriteCh1_Click(object sender, EventArgs e)
        {
            NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                    0x02,
                    (byte)this.m_comboBoxModeCh1.SelectedIndex,
                    m_pwmValueCh1,
                    (byte)this.m_HoldingTimeCh1.Value,
                    m_checkBoxSwitchCh1.Checked ? (byte)0x01 : (byte)0,
                    this.pvSetCh1PwmParamsCallback);
        }

        private void m_btnReadCh2_Click(object sender, EventArgs e)
        {
            NativeAPI.NKLC_GetPwmParams_Async(m_DevId, 0x04, this.pvGetCh2PwmParamsCallback);
        }

        private void m_btnWriteCh2_Click(object sender, EventArgs e)
        {
            NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                    0x04,
                    (byte)this.m_comboBoxModeCh2.SelectedIndex,
                    m_pwmValueCh2,
                    (byte)this.m_HoldingTimeCh2.Value,
                    m_checkBoxSwitchCh2.Checked ? (byte)0x01 : (byte)0,
                    this.pvSetCh2PwmParamsCallback);
        }

        private void m_btnReadCh3_Click(object sender, EventArgs e)
        {
            NativeAPI.NKLC_GetPwmParams_Async(m_DevId, 0x08, this.pvGetCh3PwmParamsCallback);
        }

        private void m_btnWriteCh3_Click(object sender, EventArgs e)
        {
            NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                    0x08,
                    (byte)this.m_comboBoxModeCh3.SelectedIndex,
                    m_pwmValueCh3,
                    (byte)this.m_HoldingTimeCh3.Value,
                    m_checkBoxSwitchCh3.Checked ? (byte)0x01 : (byte)0,
                    this.pvSetCh3PwmParamsCallback);
        }

        private void m_checkBoxSwitchCh0_CheckedChanged(object sender, EventArgs e)
        {
            NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                     0x01,
                     (byte)this.m_comboBoxModeCh0.SelectedIndex,
                     m_pwmValueCh0,
                     (byte)this.m_HoldingTimeCh0.Value,
                     m_checkBoxSwitchCh0.Checked ? (byte)0x01 : (byte)0,
                     this.pvSetCh0PwmParamsCallback);
        }

        private void m_checkBoxSwitchCh1_CheckedChanged(object sender, EventArgs e)
        {
            NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                    0x02,
                    (byte)this.m_comboBoxModeCh1.SelectedIndex,
                    m_pwmValueCh1,
                    (byte)this.m_HoldingTimeCh1.Value,
                    m_checkBoxSwitchCh1.Checked ? (byte)0x01 : (byte)0,
                    this.pvSetCh1PwmParamsCallback);
        }

        private void m_checkBoxSwitchCh2_CheckedChanged(object sender, EventArgs e)
        {
            NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                    0x04,
                    (byte)this.m_comboBoxModeCh2.SelectedIndex,
                    m_pwmValueCh2,
                    (byte)this.m_HoldingTimeCh2.Value,
                    m_checkBoxSwitchCh2.Checked ? (byte)0x01 : (byte)0,
                    this.pvSetCh2PwmParamsCallback);
        }

        private void m_checkBoxSwitchCh3_CheckedChanged(object sender, EventArgs e)
        {
            NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                    0x08,
                    (byte)this.m_comboBoxModeCh3.SelectedIndex,
                    m_pwmValueCh3,
                    (byte)this.m_HoldingTimeCh3.Value,
                    m_checkBoxSwitchCh3.Checked ? (byte)0x01 : (byte)0,
                    this.pvSetCh3PwmParamsCallback);
        }

        private void m_trackBarCh0_ValueChanged(object sender, EventArgs e)
        {
            m_pwmValueCh0 = (Byte)this.m_trackBarCh0.Value;
            this.m_numericUpDownCh0.Value = m_pwmValueCh0;
            if (m_checkBoxSwitchCh0.Checked)
            {
                NativeAPI.NKLC_SetPwmParams_Async(m_DevId, 
                    0x01, 
                    (byte)this.m_comboBoxModeCh0.SelectedIndex, 
                    m_pwmValueCh0, 
                    (byte)this.m_HoldingTimeCh0.Value, 
                    1, 
                    this.pvSetCh0PwmParamsCallback);
            }
            
        }
        

        private void m_numericUpDownCh0_ValueChanged(object sender, EventArgs e)
        {
            m_pwmValueCh0 = (Byte)this.m_numericUpDownCh0.Value;
            this.m_trackBarCh0.Value = m_pwmValueCh0;
        }

        private void m_trackBarCh1_ValueChanged(object sender, EventArgs e)
        {
            m_pwmValueCh1 = (Byte)this.m_trackBarCh1.Value;
            this.m_numericUpDownCh1.Value = m_pwmValueCh1;
            if (m_checkBoxSwitchCh1.Checked)
            {
                NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                    0x02,
                    (byte)this.m_comboBoxModeCh1.SelectedIndex,
                    m_pwmValueCh1,
                    (byte)this.m_HoldingTimeCh1.Value,
                    1,
                    this.pvSetCh1PwmParamsCallback);
            }
        }

        private void m_numericUpDownCh1_ValueChanged(object sender, EventArgs e)
        {
            m_pwmValueCh1 = (Byte)this.m_numericUpDownCh1.Value;
            this.m_trackBarCh1.Value = m_pwmValueCh1;
        }

        private void m_trackBarCh2_ValueChanged(object sender, EventArgs e)
        {
            m_pwmValueCh2 = (Byte)this.m_trackBarCh2.Value;
            this.m_numericUpDownCh2.Value = m_pwmValueCh2;

            if (m_checkBoxSwitchCh2.Checked)
            {
                NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                    0x04,
                    (byte)this.m_comboBoxModeCh2.SelectedIndex,
                    m_pwmValueCh2,
                    (byte)this.m_HoldingTimeCh2.Value,
                    1,
                    this.pvSetCh2PwmParamsCallback);
            }
        }

        private void m_numericUpDownCh2_ValueChanged(object sender, EventArgs e)
        {
            m_pwmValueCh2 = (Byte)this.m_numericUpDownCh2.Value;
            this.m_trackBarCh2.Value = m_pwmValueCh2;
        }

        private void m_trackBarCh3_ValueChanged(object sender, EventArgs e)
        {
            m_pwmValueCh3 = (Byte)this.m_trackBarCh3.Value;
            this.m_numericUpDownCh3.Value = m_pwmValueCh3;

            if (m_checkBoxSwitchCh3.Checked)
            {
                NativeAPI.NKLC_SetPwmParams_Async(m_DevId,
                    0x08,
                    (byte)this.m_comboBoxModeCh3.SelectedIndex,
                    m_pwmValueCh3,
                    (byte)this.m_HoldingTimeCh3.Value,
                    1,
                    this.pvSetCh3PwmParamsCallback);
            }
        }

        private void m_numericUpDownCh3_ValueChanged(object sender, EventArgs e)
        {
            m_pwmValueCh3 = (Byte)this.m_numericUpDownCh3.Value;
            this.m_trackBarCh3.Value = m_pwmValueCh3;
        }




        /// <summary>
        /// Callbacks 
        /// </summary>
        /// <param name="args"></param>

        public static void openComCallBcak( LC_CALLBACK_ARG_T args)
        {
            pThis.m_btnConnect.Text = "Disconnect";

            pThis.m_lblHardwareVer.Text = string.Format("{0}.{1}.{2}", args.openComCallbackArg.hardwareMajorVer,
                    args.openComCallbackArg.hardwareMinorVer,
                    args.openComCallbackArg.hardwareRevVer);

            pThis.m_lblFirmwareVer.Text = string.Format("{0}.{1}.{2}", args.openComCallbackArg.firmwareMajorVer,
                args.openComCallbackArg.firmwareMinorVer,
                args.openComCallbackArg.firmwareRevVer);

            // Get all of the information set when opened
            NativeAPI.NKLC_GetPwmParams_Async(pThis.m_DevId, 0x01, pThis.pvGetCh0PwmParamsCallback);
            NativeAPI.NKLC_GetPwmParams_Async(pThis.m_DevId, 0x02, pThis.pvGetCh1PwmParamsCallback);
            NativeAPI.NKLC_GetPwmParams_Async(pThis.m_DevId, 0x04, pThis.pvGetCh2PwmParamsCallback);
            NativeAPI.NKLC_GetPwmParams_Async(pThis.m_DevId, 0x08, pThis.pvGetCh3PwmParamsCallback);
        }


        public static void closeComCallBack(LC_CALLBACK_ARG_T args)
        {
            pThis.m_btnConnect.Text = "Connect";
            pThis.m_lblFirmwareVer.Text = "x.x.x";
            pThis.m_lblHardwareVer.Text = "x.x.x";
        }
        public static void getDeviceVerCallBack( LC_CALLBACK_ARG_T args)
        {
            if (args.getDeviceVerCallbackArg.error > 0)
            {
                //m_lblHardwareVer.Text = string.Format()
                MessageBox.Show(pThis, "Get device num error");
                pThis.m_lblFirmwareVer.Text = "x.x.x";
                pThis.m_lblHardwareVer.Text = "x.x.x";
            }
            else
            {
                pThis.m_lblHardwareVer.Text = string.Format("{0}.{1}.{2}", args.getDeviceVerCallbackArg.hardwareMajorVer,
                    args.getDeviceVerCallbackArg.hardwareMinorVer,
                    args.getDeviceVerCallbackArg.hardwareRevVer);

                pThis.m_lblFirmwareVer.Text = string.Format("{0}.{1}.{2}",args.getDeviceVerCallbackArg.firmwareMajorVer,
                    args.getDeviceVerCallbackArg.firmwareMinorVer,
                    args.getDeviceVerCallbackArg.firmwareRevVer);
            }

        }
        public static void setPwmParamsCallBackCh0( LC_CALLBACK_ARG_T args)
        {
            if (args.setPwmParamsCallbackArg.error > 0)
            {
                MessageBox.Show(pThis, "Set params to CH0 Error");
            }
        }
        public static void getPwmParamsCallBackCh0( LC_CALLBACK_ARG_T args)
        {
            if (args.getPwmParamsCallbackArg.error > 0)
            {
                MessageBox.Show(pThis, "getPwmParamsCh0 Error");
            }
            else
            {
                pThis.m_checkBoxSwitchCh0.Checked = args.getPwmParamsCallbackArg.pwmOnOff > 0? true: false;
                pThis.m_trackBarCh0.Value = args.getPwmParamsCallbackArg.pwmValue;
                pThis.m_comboBoxModeCh0.SelectedIndex = args.getPwmParamsCallbackArg.pwmMode;
                pThis.m_HoldingTimeCh0.Value = args.getPwmParamsCallbackArg.pwmHoldingTime;
            }
        }

        public static void setPwmParamsCallBackCh1(LC_CALLBACK_ARG_T args)
        {
            if (args.setPwmParamsCallbackArg.error > 0)
            {
                MessageBox.Show(pThis, "Set params to CH1 Error");
            }
        }
        public static void getPwmParamsCallBackCh1(LC_CALLBACK_ARG_T args)
        {
            if (args.getPwmParamsCallbackArg.error > 0)
            {
                MessageBox.Show(pThis, "getPwmParamsCh1 Error");
            }
            else
            {
                pThis.m_checkBoxSwitchCh1.Checked = args.getPwmParamsCallbackArg.pwmOnOff > 0 ? true : false;
                pThis.m_trackBarCh1.Value = args.getPwmParamsCallbackArg.pwmValue;
                pThis.m_comboBoxModeCh1.SelectedIndex = args.getPwmParamsCallbackArg.pwmMode;
                pThis.m_HoldingTimeCh1.Value = args.getPwmParamsCallbackArg.pwmHoldingTime;
            }
        }

        public static void setPwmParamsCallBackCh2(LC_CALLBACK_ARG_T args)
        {
            if (args.setPwmParamsCallbackArg.error > 0)
            {
                MessageBox.Show(pThis, "Set params to CH2 Error");
            }
        }
        public static void getPwmParamsCallBackCh2(LC_CALLBACK_ARG_T args)
        {
            if (args.getPwmParamsCallbackArg.error > 0)
            {
                MessageBox.Show(pThis, "getPwmParamsCh2 Error");
            }
            else
            {
                pThis.m_checkBoxSwitchCh2.Checked = args.getPwmParamsCallbackArg.pwmOnOff > 0 ? true : false;
                pThis.m_trackBarCh2.Value = args.getPwmParamsCallbackArg.pwmValue;
                pThis.m_comboBoxModeCh2.SelectedIndex = args.getPwmParamsCallbackArg.pwmMode;
                pThis.m_HoldingTimeCh2.Value = args.getPwmParamsCallbackArg.pwmHoldingTime;
            }
        }

        public static void setPwmParamsCallBackCh3(LC_CALLBACK_ARG_T args)
        {
            if (args.setPwmParamsCallbackArg.error > 0)
            {
                MessageBox.Show(pThis, "Set params to CH3 Error");
            }
        }
        public static void getPwmParamsCallBackCh3(LC_CALLBACK_ARG_T args)
        {
            if (args.getPwmParamsCallbackArg.error > 0)
            {
                MessageBox.Show(pThis, "getPwmParamsCh3 Error");
            }
            else
            {
                pThis.m_checkBoxSwitchCh3.Checked = args.getPwmParamsCallbackArg.pwmOnOff > 0 ? true : false;
                pThis.m_trackBarCh3.Value = args.getPwmParamsCallbackArg.pwmValue;
                pThis.m_comboBoxModeCh3.SelectedIndex = args.getPwmParamsCallbackArg.pwmMode;
                pThis.m_HoldingTimeCh3.Value = args.getPwmParamsCallbackArg.pwmHoldingTime;
            }
        }

    }
}
