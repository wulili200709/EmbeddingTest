using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;


namespace NK_IO_LC_TEST_CSharp
{

    public partial class LoginForm : Form
    {
        string configFilePath = "";
        string selectFilePath = "";
        public LoginForm()
        {
            InitializeComponent();
            configFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "config.ini");//在当前程序路径创建
            selectFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "select.ini");

            List<string> stringList = INIHelper.ReadSections(configFilePath);
            this.comboBox1.DataSource = stringList;

            string selectItem = INIHelper.Read("SELECTED", "Name", " ", selectFilePath);
            this.comboBox1.SelectedItem = selectItem;

        }

        private void m_btnExit_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.Cancel;
        }

        private void m_btnAccept_Click(object sender, EventArgs e)
        {
            INIHelper.Write("SELECTED", "Name", this.comboBox1.SelectedItem.ToString(), selectFilePath);
            string selectCfgPath = INIHelper.Read(this.comboBox1.SelectedItem.ToString(), "ConfigPath", "", configFilePath);
            INIHelper.Write("SELECTED", "ConfigPath", selectCfgPath, selectFilePath);
            this.DialogResult = DialogResult.OK;
            this.Close();
        }
    }
}
