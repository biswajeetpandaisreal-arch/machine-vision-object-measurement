#include "unified.h"
#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

int main()
{
    cout << "==============================" << endl;
    cout << "   MEASUREMENT SYSTEM" << endl;
    cout << "==============================" << endl;
    cout << "Select mode:" << endl;
    cout << "  1 = Live Camera" << endl;
    cout << "  2 = Image File" << endl;
    cout << "------------------------------" << endl;
    cout << "Enter choice (1 or 2): ";

    int choice;
    cin >> choice;
    cin.ignore();   // clears leftover newline from buffer

    if (choice == 1)
    {
        cout << "Starting live camera..." << endl;
        runUnifiedSystem();
    }
    else if (choice == 2)
    {
        cout << "Enter full image path" << endl;
        cout << "(e.g. C:/Users/Name/test.jpg): " << endl;
        cout << "Path: ";

        string path;
        getline(cin, path);   // reads full line including spaces

        // Remove surrounding quotes if user added them
        if (!path.empty() && path.front() == '"') path.erase(0, 1);
        if (!path.empty() && path.back() == '"') path.pop_back();

        // Replace backslashes with forward slashes
        replace(path.begin(), path.end(), '\\', '/');

        cout << "Loading image: " << path << endl;
        runUnifiedSystem(path);
    }
    else
    {
        cout << "Invalid choice. Defaulting to live camera." << endl;
        runUnifiedSystem();
    }

    return 0;
}
